from preprocessing.filter_index import PodcastDatabase
from preprocessing.extract_episodes import EpisodeFetcher
import pandas as pd
import numpy as np

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
    def __init__(self, pod_db: PodcastDatabase, ep_fetcher: EpisodeFetcher):
        self.pod_db = pod_db
        self.ep_fetcher = ep_fetcher
        self.rng = np.random.default_rng(SEED)

    def build_podcast_df(self, k: int) -> pd.DataFrame: # k corresponds to num podcasts per year
        try:
            feed_urls = self.pod_db.filter(filters=filters)
            print(f"Gefundene Podcasts: {len(feed_urls)}")
        finally:
            self.pod_db.close()
        
        df = pd.DataFrame(self.ep_fetcher.fetch_episodes_for_podcasts(feed_urls=feed_urls))
        df = df.dropna(subset=["episode_year", "episode_audio_url", "episode_length"])
        length_mask = df["episode_length"] >= MIN_PODCAST_LEN_MINUTES * 60
        df = df[length_mask]
        year_mask = df["episode_year"].isin(YEARS) # filter out episodes that are not in the desired years
        df = df[year_mask]
        counts = df.groupby(["feed_url", "episode_year"]).size().unstack(fill_value=0).reindex(columns=YEARS, fill_value=0) # pd series with counts num of episodes with each covered year and fill up with zeros 
        keep = counts.index[(counts >= k).all(axis=1)] # keep only podcasts with counts >= k along columns (years)
        df = df[df["feed_url"].isin(keep)]
        return df

    def draw_random_podcasts(self, df:pd.DataFrame, num_p:int) -> pd.DataFrame:
        feed_urls = df["feed_url"].unique()
        randomized_pds = self.rng.choice(feed_urls, size=num_p, replace=False)
        return df[df["feed_url"].isin(randomized_pds)]

    def draw_random_episodes(self, df:pd.DataFrame, k:int) -> pd.DataFrame:
        return df.groupby(["feed_url"], ["episode_year"]).sample(n=k, random_state=SEED) # Draw k random podcasts for each year per podcast
