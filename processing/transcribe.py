from pathlib import Path
import json
import pandas as pd
import logging
import os
import dotenv
from deepgram import (
    DeepgramClient,
)

TRANSCRIPTION_DIR = Path(__file__).resolve().parent.parent / "transcriptions"

class EpisodeTranscriber():
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        dotenv.load_dotenv()
        self.deepgram = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))

    def transcribe(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def transcribe_one(self, episode_hash: str, episode_path: Path):
        out = self.out_dir / f"{episode_hash}.json"
        if out.exists():
            return episode_hash, True, str(out), None
        tmp = out.with_suffix(".part.json")

        try:
            logging.info(f"Starting transcription for {episode_hash}")
            with open(episode_path, "rb") as audio_file:
                response = self.deepgram.listen.v1.media.transcribe_file(
                    request=audio_file.read(),
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
            return episode_hash, False, None, repr(e)