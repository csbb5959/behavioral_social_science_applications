import pandas as pd
from download import EpisodeDownloader

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

class ProcessingPipeline():
    def __init__(self, ep_downloader: EpisodeDownloader):
        self.ep_downloader = ep_downloader

    def process(self, df: pd.DataFrame, k_test:int): # k_test corresponds to amount of episodes we test per year 
        """Full process: Donwload -> Transcription -> Spontaneity Check -> Ad Removal
        At first, we only take k_test episodes for the first and last year to check scripted vs. non scripted, before processing
        k episodes per year for each podcast."""

        df["fail_stage"] = pd.NA # Safe stage in case of failure
        df["fail_reason"] = pd.NA # Safe reason in case of failure

        # 1. Download
        # 1.1. Test run with k_test first and last year episodes
        df["audio_path"] = pd.NA 
        df["processing_status"] = "pending"
        df_test = self.run_download(df=df, years=[max(YEARS), min(YEARS)], k_test=k_test) # Initial run with first and last year episode

    def run_download(self, df, years:list, k_test: int) -> pd.DataFrame:
        while True:
            candidate = self.__screening(df[(df["processing_status"] != "failed")], years=years, k_test=k_test)
            todo = candidate[candidate["audio_path"].isna()]
            if todo.empty:
                return df

            results = self.ep_downloader.download(todo)

            for result in results.itertuples(index=False): 
                row = df["episode_hash"] == result.episode_hash
                if result.success:
                    df.loc[row, "audio_path"] = result.path
                else:
                    df.loc[row, "processing_status"] = "failed"
                    df.loc[row, "fail_stage"] = "download"
                    df.loc[row, "fail_reason"] = result.error

            return df

    def __screening(self, df, years:list, k_test: int) -> pd.DataFrame: 
        first_last = df[df["episode_year"].isin(years)] # only episodes from desired years
        return first_last.groupby(["feed_url", "episode_year"]).head(k_test)

    def transcribe(self, df: pd.DataFrame, )
        
