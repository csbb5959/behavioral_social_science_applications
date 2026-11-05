from filter_index import PodcastDatabase
from extract_episodes import EpisodeFetcher
import pandas as pd
import numpy as np
import hashlib

DATABASE_PATH = "data/podcastindex_feeds.db"
TABLE = "podcasts"
YEARS = [2022, 2023, 2024, 2025, 2026]
filters = {"language": ["en", "en-us"], 
               "category1": ["games", "science"], 
               "category2": ["hobbies", "medicine"], 
               "category3": ["video games", "education"],
               "newestItemPubdate": [2026],
               "oldestItemPubdate": [2018],
               "dead": ["0"]}
MIN_PODCAST_LEN_MINUTES = 15
SEED = 23

class FetchPipeline():
    def __init__(self, pod_db: PodcastDatabase, ep_fetcher: EpisodeFetcher, group: str, filters: dict):
        self.pod_db = pod_db
        self.ep_fetcher = ep_fetcher
        self.rng = np.random.default_rng(SEED)
        self.group = group
        self.filters = filters

    def build_podcast_df(self, k: int, buffer: int) -> pd.DataFrame: # k corresponds to num podcasts per year
        try:
            feed_urls = self.pod_db.filter(filters=self.filters)
            print(f"Num podcasts found: {len(feed_urls)}")
        finally:
            self.pod_db.close()
        
        df = pd.DataFrame(self.ep_fetcher.fetch_episodes_for_podcasts(feed_urls=feed_urls))
        df = df.dropna(subset=["episode_year", "episode_audio_url", "episode_length"])
        length_mask = df["episode_length"] >= MIN_PODCAST_LEN_MINUTES * 60
        df = df[length_mask]
        year_mask = df["episode_year"].isin(YEARS) # filter out episodes that are not in the desired years
        df = df[year_mask]
        counts = df.groupby(["feed_url", "episode_year"]).size().unstack(fill_value=0).reindex(columns=YEARS, fill_value=0) # pd series with counts num of episodes with each covered year and fill up with zeros 
        keep = counts.index[(counts >= buffer * k).all(axis=1)] # keep only podcasts with counts >= k along columns (years)
        df = df[df["feed_url"].isin(keep)]
        df["group"] = self.group
        df = self.__add_hash_podcasts_rank(df)
        df = self.__add_episode_hash_rank(df)
        df["processing_status"] = "pending"
        df = self.__sort(df)
        return df

    def __select(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        return df.groupby(["feed_url", "year"]).head(k)

    def __sort(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(by=["podcast_hash", "episode_year", "episode_hash"])

    def __add_hash_podcasts_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        df["podcast_hash"] = df["feed_url"].map(self.__rank)
        return df

    def __add_episode_hash_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        df["episode_hash"] = df["episode_audio_url"].map(self.__rank)
        return df

    def __rank(self, key: str) -> str:
        return hashlib.sha256((f"{SEED}:{key}").encode("utf-8")).hexdigest()
