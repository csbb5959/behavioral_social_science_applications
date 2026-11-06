import logging
import pandas as pd
import requests

DOWNLOAD_FOLDER = "../downloads/"

class PodcastDownloadError(Exception):
    """Raised when a podcast episode fails to download."""
    pass

class EpisodeDownloader():
    
    def download_episode(self, audio_url: str, episode_name: str, chunk_size=1024):
        episode_mp3 = requests.get(audio_url, stream=True)

        download_path = DOWNLOAD_FOLDER + episode_name 

        if episode_mp3.status_code == 200:
            with open(download_path, 'wb') as f:
                for chunk in episode_mp3.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
            logging.info(f"episode successfully saved in file {download_path}")
        else:
            raise PodcastsDownloadError(f"episode {episode_name} could not be downloaded: {episode_mp3.status_code}")