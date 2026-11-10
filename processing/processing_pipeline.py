import logging
import pandas as pd
from pandas import Series
from download import EpisodeDownloader
from transcribe import EpisodeTranscriber

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

class ProcessingPipeline():
    def __init__(self, ep_downloader: EpisodeDownloader, ep_transcriber: EpisodeTranscriber):
        self.ep_downloader = ep_downloader
        self.ep_transcriber = ep_transcriber

    def process(self, df: pd.DataFrame, k_test: int, n_podcasts: int) -> pd.DataFrame: # k_test: episodes tested per year, n_podcasts: podcasts per group
        """Full process: Download -> Transcription -> Ad Removal -> Spontaneity Check
        At first, we only take k_test episodes for the first and last year to check scripted vs. non scripted, before processing
        k episodes per year for each podcast."""

        df["fail_stage"] = pd.NA # Safe stage in case of failure
        df["episode_fail_reason"] = pd.NA # Safe reason in case of failure
        df["audio_path"] = pd.NA # Path for audio wav file
        df["episode_status"] = "pending" # Track status
        df["transcript_path"] = pd.NA # Path for transcription json file
        df["podcast_status"] = "pending" # Podcast-level status, "failed" excludes the whole podcast
        df["podcast_fail_reason"] = pd.NA

        # 1. Test -- with episodes from first and last year until after spontaneity check
        df = self.run_screening(df=df, years=[min(YEARS), max(YEARS)], k=k_test, n_podcasts=n_podcasts)
        return df

    # Both selections rely on df being sorted by rank, as returned by FetchPipeline.build_podcast_df
    def __select_podcasts(self, df: pd.DataFrame, n_podcasts: int) -> Series:
        active = df[df["podcast_status"] != "failed"]
        first_n = active.drop_duplicates("feed_url").groupby("group").head(n_podcasts) # first n podcasts in rank order per group
        return first_n["feed_url"]

    def __select_episodes(self, df: pd.DataFrame, podcasts: Series, years: list, k: int) -> pd.DataFrame:
        active = df[(df["episode_status"] != "failed") & df["feed_url"].isin(podcasts) & df["episode_year"].isin(years)]
        return active.groupby(["feed_url", "episode_year"]).head(k) # first k episodes in rank order per podcast and year

    def __incomplete_podcasts(self, candidates: pd.DataFrame, podcasts: Series, years: list, k: int) -> list:
        """Podcasts that have fewer than k remaining episodes in at least one year during screening."""
        counts = (candidates.groupby(["feed_url", "episode_year"]).size()
                  .unstack(fill_value=0)
                  .reindex(index=podcasts.tolist(), columns=years, fill_value=0))
        return counts.index[(counts < k).any(axis=1)].tolist()

    def __warn_if_too_few(self, df: pd.DataFrame, podcasts: Series, n_podcasts: int):
        per_group = df[df["feed_url"].isin(podcasts)].groupby("group")["feed_url"].nunique()
        per_group = per_group.reindex(df["group"].unique(), fill_value=0)
        for group, count in per_group[per_group < n_podcasts].items():
            logging.warning(f"Group '{group}': only {count} of {n_podcasts} podcasts left after screening")

    def run_screening(self, df: pd.DataFrame, years: list, k: int, n_podcasts: int) -> pd.DataFrame:
        while True:
            podcasts = self.__select_podcasts(df, n_podcasts)
            candidates = self.__select_episodes(df, podcasts, years, k) # max k non-failed episodes per podcast and year

            incomplete = self.__incomplete_podcasts(candidates, podcasts, years, k)
            if incomplete:
                rows = df["feed_url"].isin(incomplete)
                df.loc[rows, "podcast_status"] = "failed"
                df.loc[rows, "podcast_episode_fail_reason"] = "not enough episodes"
                continue                           # next podcast in rank order moves up

            if candidates["transcript_path"].notna().all():
                self.__warn_if_too_few(df, podcasts, n_podcasts)
                return df                          # every selected episode passed all stages

            hashes = candidates["episode_hash"]
            df = self.__download(df, hashes)       # only those without audio_path
            df = self.__transcribe(df, hashes)     # only those with audio_path but without transcript_path


    def __download(self, df:pd.DataFrame, hashes:Series) -> pd.DataFrame:
        todo = df[df["episode_hash"].isin(hashes) & df["audio_path"].isna() & df["transcript_path"].isna()]
        results = self.ep_downloader.download(todo)

        for result in results.itertuples(index=False): 
            row = df["episode_hash"] == result.episode_hash
            if result.success:
                df.loc[row, "audio_path"] = result.path
                df.loc[row, "episode_status"] = "downloaded"
            else:
                df.loc[row, "episode_status"] = "failed"
                df.loc[row, "fail_stage"] = "download"
                df.loc[row, "episode_fail_reason"] = result.error

        return df

    def __transcribe(self, df:pd.DataFrame, hashes:Series) -> pd.DataFrame:
        todo = df[df["episode_hash"].isin(hashes) & df["audio_path"].notna() & df["transcript_path"].isna()]
        results = self.ep_transcriber.transcribe(todo)

        for result in results.itertuples(index=False):
            row = df["episode_hash"] == result.episode_hash
            if result.success:
                df.loc[row, "transcript_path"] = result.path
                df.loc[row, "episode_status"] = "transcribed"
            else:
                df.loc[row, "episode_status"] = "failed"
                df.loc[row, "fail_stage"] = "transcription"
                df.loc[row, "episode_fail_reason"] = result.error

        return df
        
