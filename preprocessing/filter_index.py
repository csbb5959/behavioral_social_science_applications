import sqlite3
from time import timezone
from datetime import datetime, timezone

DATABASE_PATH = "data/podcastindex_feeds.db"

class PodcastDatabase():
    def __init__(self, db_path: str, table: str):
        self._con = sqlite3.connect(db_path)
        self._cur = self._con.cursor()
        self._table = table

    def get_columns (self):
        return self._cur.execute(f"PRAGMA table_info({self._table})").fetchall()       

    def filter(self, filters: dict, limit=20) -> list:
        query = f"SELECT url FROM {self._table} WHERE 1=1"
        params = []

        for column, values in filters.items():
            if values: 
                placeholders = ",".join(["?"] * len(values))
                if column == "newestItemPubdate": 
                    query += f" AND {column} > (?)" # Filter out podcasts whose newest Item update is older than filter year
                    params.append(self.__date_to_utc_int(year=values[0])) # timezone object needs to be casted into a unix int for db comparison
                elif column == "oldestItemPubdate": 
                    query += f" AND {column} < (?)" # Filter out podcasts whose oldest Item update is newer than filter year
                    params.append(self.__date_to_utc_int(year=values[0])) # timezone object needs to be casted into a unix int for db comparison
                else:
                    query += f" AND {column} IN ({placeholders})"
                    params.extend(values)  

        feed_url_tupels = self._cur.execute(query + f"LIMIT {limit}", params).fetchall()
        return self.__process_to_str_list(feed_url_tupels)

    def __date_to_utc_int(self, year: int, month=1, day=1) -> int:
        dt = datetime(year=year, month=month, day=day, tzinfo=timezone.utc)
        return int(dt.timestamp())

    def __process_to_str_list(self, feed_url_tupels):
        return ["".join(feed) for feed in feed_url_tupels]

    def close(self):
        self._con.close()


if __name__ == "__main__":

    filters = {"language": ["en", "en-us"], 
               "category1": ["games", "science"], 
               "category2": ["hobbies", "medicine"], 
               "category3": ["video games", "education"],
               "newestItemPubdate": [2026],
               "oldestItemPubdate": [2018],
               "dead": ["0"]}
    
    pod_db = PodcastDatabase(DATABASE_PATH, table="podcasts")

    filtered_pods = pod_db.filter(filters=filters, limit=10)
    pod_db.close()
    print(filtered_pods)
    

    