import feedparser
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

class EpisodeFetcher():
    def fetch_episodes_for_podcasts(self, feed_urls, max_workers=16):
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            rows = [r for result in ex.map(self.__fetch_episodes, feed_urls) for r in result]
        return rows

    def __fetch_episodes(self, feed_url):
            try:
                parsed = feedparser.parse(feed_url)
            except Exception:
                return []
            return [
                {
                    "feed_url": feed_url,
                    "podcast_title": parsed.feed.title,
                    "episode_title": entry["title"],
                    "episode_year": entry["published_parsed"].tm_year,
                    "episode_audio_url": entry["enclosures"][0]["href"],
                    "episode_length": self.__to_seconds(entry.get("itunes_duration", None))
                }
                for entry in parsed.entries
                if entry["published_parsed"] and entry["enclosures"]
            ]

    def __to_seconds(self, raw):
        try:
            secs = 0
            for part in raw.split(":"):   # "1:02:03" -> 3723, "45:30" -> 2730, "3600" -> 3600
                secs = secs * 60 + int(float(part))
            return secs or None           # 0 wird zu None
        except (AttributeError, ValueError):   # None oder kein Zahlenformat
            return None

