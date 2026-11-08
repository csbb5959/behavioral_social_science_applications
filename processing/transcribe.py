from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import logging
import os
import dotenv
from deepgram.core.api_error import ApiError
from deepgram import (
    DeepgramClient,
)

TRANSCRIPTION_DIR = Path(__file__).resolve().parent.parent / "transcriptions"
MAX_RETRIES = 2

class EpisodeTranscriber():
    def __init__(self, out_dir: Path, max_workers: int=4):
        self.out_dir = out_dir
        self.max_workers = max_workers
        self.out_dir.mkdir(parents=True, exist_ok=True)
        dotenv.load_dotenv()
        self.deepgram = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))

    def transcribe(self, df: pd.DataFrame) -> pd.DataFrame:
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            results = list(ex.map(self.__transcribe_one, df["episode_hash"], df["audio_path"]))
            return pd.DataFrame(results, columns=["episode_hash", "success", "path", "error"])

    def __transcribe_one(self, episode_hash: str, episode_path: Path):
        out = self.out_dir / f"{episode_hash}.json"
        if out.exists():
            return episode_hash, True, str(out), None
        tmp = out.with_suffix(".part.json")

        try:
            logging.info(f"Starting transcription for {episode_hash}")
            with open(episode_path, "rb") as audio_file:
                response = self.deepgram.listen.v1.media.transcribe_file(
                    request=audio_file.read(),
                    request_options={"max_retries": MAX_RETRIES},
                    model="nova-3",
                    smart_format=True,
                )
        
            with open(tmp, "w", encoding='utf-8') as transcript_file:
                logging.info(f"Writing to temp file {tmp}")
                transcript_file.write(response.model_dump_json(indent=4))
            
            tmp.rename(out)
            return episode_hash, True, str(out), None
        
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            tmp.unlink(missing_ok=True)
            err = f"ApiError {e.status_code}: {e.body}" if isinstance(e, ApiError) else repr(e)
            return episode_hash, False, None, err