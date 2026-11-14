import logging
import pandas as pd
from pandas import Series
from download import EpisodeDownloader
from transcribe import EpisodeTranscriber
from remove_ads import AdRemover
from classify_spontaneity import SpontaneityClassifier
from config import YEARS, SCREENING_YEARS, MIN_WORDS_PER_EPISODE

class ProcessingPipeline():
    """Runs download, transcription, ad removal and spontaneity check on the episode table.
    Failed episodes and podcasts are replaced by the next one in rank order."""
    def __init__(self, ep_downloader: EpisodeDownloader, ep_transcriber: EpisodeTranscriber,
                 ad_remover: AdRemover, classifier: SpontaneityClassifier):
        """Takes one object per stage."""
        self.ep_downloader = ep_downloader
        self.ep_transcriber = ep_transcriber
        self.ad_remover = ad_remover
        self.classifier = classifier

    def process(self, df: pd.DataFrame, k: int, n_podcasts: int, k_test: int = 1) -> pd.DataFrame:
        """Full process: Download -> Transcription -> Ad Removal -> Spontaneity Check
        k: episodes per podcast and year, n_podcasts: podcasts per group, k_test: screening episodes per screening year.
        At first, only k_test episodes from each screening year are processed to check scripted vs. non scripted.
        Only podcasts that pass get k episodes per year processed."""

        new_columns = {
            "episode_status": "pending",    # pending -> downloaded -> transcribed -> cleaned -> passed, or failed
            "fail_stage": pd.NA,            # Stage in case of failure
            "episode_fail_reason": pd.NA,   # Reason in case of failure
            "audio_path": pd.NA,            # Path for audio wav file
            "transcript_path": pd.NA,       # Path for transcription json file
            "cleaned_path": pd.NA,          # Path for json file with ad marks
            "n_words": pd.NA,               # Words left after ad removal
            "n_speakers": pd.NA,            # Number of speakers (after ad removal)
            "main_speaker_share": pd.NA,    # Share of words spoken by the main speaker (after ad removal)
            "mean_confidence": pd.NA,       # Mean word confidence of the transcription (after ad removal)
            "score": pd.NA,                 # Spontaneity score 1-4
            "scripted": pd.NA,              # True if score is below the threshold
            "scripted_reason": pd.NA,       # One-sentence reason from the LLM
            "llm_model": pd.NA,             # Model that gave the score
            "podcast_status": "pending",    # pending -> passed (screening), or failed (excludes the whole podcast)
            "podcast_fail_reason": pd.NA,
        }
        for column, value in new_columns.items():
            if column not in df.columns:    # keep existing columns, so a saved df can be continued
                df[column] = value

        while True:
            df = self.run_screening(df, years=SCREENING_YEARS, k=k_test, n_podcasts=n_podcasts)
            df, complete = self.run_main(df, years=YEARS, k=k, n_podcasts=n_podcasts)
            if complete:
                break                       # otherwise a podcast failed and its replacement needs screening first

        self.__warn_if_too_few(df, self.__select_podcasts(df, n_podcasts), n_podcasts)
        return df

    def run_screening(self, df: pd.DataFrame, years: list, k: int, n_podcasts: int) -> pd.DataFrame:
        """Runs until the first n_podcasts non-failed podcasts per group have passed the spontaneity check."""
        while True:
            podcasts = self.__select_podcasts(df, n_podcasts)
            candidates = self.__select_episodes(df, podcasts, years, k) # max k non-failed episodes per podcast and year

            incomplete = self.__incomplete_podcasts(candidates, podcasts, years, k)
            if incomplete:
                df = self.__fail_podcasts(df, incomplete, "not enough episodes")
                continue                            # next podcast in rank order moves up

            scripted = candidates.loc[candidates["scripted"].isin([True]), "feed_url"].unique().tolist()
            if scripted:
                df = self.__fail_podcasts(df, scripted, "scripted")
                continue                            # next podcast in rank order moves up

            if (candidates["episode_status"] == "passed").all():
                df.loc[df["feed_url"].isin(podcasts), "podcast_status"] = "passed"
                return df                           # every screening episode passed all stages and none is scripted

            df = self.__run_stages(df, candidates["episode_hash"])

    def run_main(self, df: pd.DataFrame, years: list, k: int, n_podcasts: int) -> tuple:
        """Processes k episodes per year for the screened podcasts. Scripted episodes are kept here, the score is only recorded.
        Returns (df, False) as soon as a podcast fails, so that its replacement can be screened first."""
        while True:
            podcasts = self.__select_podcasts(df, n_podcasts)
            candidates = self.__select_episodes(df, podcasts, years, k)

            incomplete = self.__incomplete_podcasts(candidates, podcasts, years, k)
            if incomplete:
                df = self.__fail_podcasts(df, incomplete, "not enough episodes")
                return df, False

            if (candidates["episode_status"] == "passed").all():
                return df, True                     # every cell (podcast x year) has k passed episodes

            df = self.__run_stages(df, candidates["episode_hash"])

    def __run_stages(self, df: pd.DataFrame, hashes: Series) -> pd.DataFrame:
        """One pass over the candidates. Each stage only handles the episodes that are waiting for it."""
        df = self.__download(df, hashes)        # pending -> downloaded
        df = self.__transcribe(df, hashes)      # downloaded -> transcribed
        df = self.__remove_ads(df, hashes)      # transcribed -> cleaned
        df = self.__classify(df, hashes)        # cleaned -> passed
        return df

    # Both selections rely on df being sorted by rank, as returned by FetchPipeline.build_podcast_df
    def __select_podcasts(self, df: pd.DataFrame, n_podcasts: int) -> Series:
        """Return the feed URLs of the first n_podcasts podcasts per group that have not failed."""
        active = df[df["podcast_status"] != "failed"]
        first_n = active.drop_duplicates("feed_url").groupby("group").head(n_podcasts) # first n podcasts in rank order per group
        return first_n["feed_url"]

    def __select_episodes(self, df: pd.DataFrame, podcasts: Series, years: list, k: int) -> pd.DataFrame:
        """Return the first k non-failed episodes per podcast and year for the given podcasts and years."""
        active = df[(df["episode_status"] != "failed") & df["feed_url"].isin(podcasts) & df["episode_year"].isin(years)]
        return active.groupby(["feed_url", "episode_year"]).head(k) # first k episodes in rank order per podcast and year

    def __incomplete_podcasts(self, candidates: pd.DataFrame, podcasts: Series, years: list, k: int) -> list:
        """Podcasts that have fewer than k remaining episodes in at least one year."""
        counts = (candidates.groupby(["feed_url", "episode_year"]).size()
                  .unstack(fill_value=0)
                  .reindex(index=podcasts.tolist(), columns=years, fill_value=0))
        return counts.index[(counts < k).any(axis=1)].tolist()

    def __fail_podcasts(self, df: pd.DataFrame, feed_urls: list, reason: str) -> pd.DataFrame:
        """Mark all episodes of the given podcasts with podcast_status "failed" and the reason."""
        rows = df["feed_url"].isin(feed_urls)
        df.loc[rows, "podcast_status"] = "failed"
        df.loc[rows, "podcast_fail_reason"] = reason
        logging.info(f"{len(feed_urls)} podcast(s) failed: {reason}")
        return df

    def __fail_episode(self, df: pd.DataFrame, row: Series, stage: str, reason) -> pd.DataFrame:
        """Mark one episode as failed and save the stage and the reason."""
        df.loc[row, "episode_status"] = "failed"
        df.loc[row, "fail_stage"] = stage
        df.loc[row, "episode_fail_reason"] = reason
        return df

    def __warn_if_too_few(self, df: pd.DataFrame, podcasts: Series, n_podcasts: int):
        """Log a warning for every group that ends up with fewer than n_podcasts podcasts."""
        per_group = df[df["feed_url"].isin(podcasts)].groupby("group")["feed_url"].nunique()
        per_group = per_group.reindex(df["group"].unique(), fill_value=0)
        for group, count in per_group[per_group < n_podcasts].items():
            logging.warning(f"Group '{group}': only {count} of {n_podcasts} podcasts left after screening")

    def __download(self, df: pd.DataFrame, hashes: Series) -> pd.DataFrame:
        """Download the candidates with status "pending" and write the result to df."""
        todo = df[df["episode_hash"].isin(hashes) & (df["episode_status"] == "pending")]
        results = self.ep_downloader.download(todo)

        for result in results.itertuples(index=False):
            row = df["episode_hash"] == result.episode_hash
            if result.success:
                df.loc[row, "audio_path"] = result.path
                df.loc[row, "episode_status"] = "downloaded"
            else:
                df = self.__fail_episode(df, row, "download", result.error)

        return df

    def __transcribe(self, df: pd.DataFrame, hashes: Series) -> pd.DataFrame:
        """Transcribe the candidates with status "downloaded" and write the result to df."""
        todo = df[df["episode_hash"].isin(hashes) & (df["episode_status"] == "downloaded")]
        results = self.ep_transcriber.transcribe(todo)

        for result in results.itertuples(index=False):
            row = df["episode_hash"] == result.episode_hash
            if result.success:
                df.loc[row, "transcript_path"] = result.path
                df.loc[row, "episode_status"] = "transcribed"
            else:
                df = self.__fail_episode(df, row, "transcription", result.error)

        return df

    def __remove_ads(self, df: pd.DataFrame, hashes: Series) -> pd.DataFrame:
        """Mark ads for the candidates with status "transcribed" and write the result to df.
        Episodes with fewer than MIN_WORDS_PER_EPISODE words left fail."""
        todo = df[df["episode_hash"].isin(hashes) & (df["episode_status"] == "transcribed")]
        results = self.ad_remover.remove_podcasts_ads(todo)

        for result in results.itertuples(index=False):
            row = df["episode_hash"] == result.episode_hash
            if not result.success:
                df = self.__fail_episode(df, row, "ad_removal", result.error)
                continue

            df.loc[row, "cleaned_path"] = result.path
            n_words = int(result.n_words) # type: ignore
            df.loc[row, "n_words"] = n_words
            df.loc[row, "n_speakers"] = result.n_speakers
            df.loc[row, "main_speaker_share"] = result.main_speaker_share
            df.loc[row, "mean_confidence"] = result.mean_confidence
            if n_words < MIN_WORDS_PER_EPISODE:
                df = self.__fail_episode(df, row, "ad_removal", f"only {n_words} words left (minimum {MIN_WORDS_PER_EPISODE})")
            else:
                df.loc[row, "episode_status"] = "cleaned"

        return df

    def __classify(self, df: pd.DataFrame, hashes: Series) -> pd.DataFrame:
        """Rate the spontaneity of the candidates with status "cleaned" and write the result to df."""
        todo = df[df["episode_hash"].isin(hashes) & (df["episode_status"] == "cleaned")]
        results = self.classifier.classify(todo)

        for result in results.itertuples(index=False):
            row = df["episode_hash"] == result.episode_hash
            if result.success:
                df.loc[row, "score"] = result.score
                df.loc[row, "scripted"] = result.scripted
                df.loc[row, "scripted_reason"] = result.scripted_reason
                df.loc[row, "llm_model"] = result.model
                df.loc[row, "episode_status"] = "passed"
            else:
                df = self.__fail_episode(df, row, "spontaneity", result.error)

        return df
