from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import json
import logging
import os
import dotenv
from deepgram.core.api_error import ApiError
from deepgram import (
    DeepgramClient,
)
from config import TRANSCRIPTION_MODEL

MAX_RETRIES = 2

class EpisodeTranscriber():
    """Transcribes audio files with Deepgram and saves the full JSON response per episode."""
    def __init__(self, out_dir: Path, max_workers: int=4):
        """Create the output folder and the Deepgram client (DEEPGRAM_API_KEY from .env). max_workers is the number of parallel requests."""
        self.out_dir = out_dir
        self.max_workers = max_workers
        self.out_dir.mkdir(parents=True, exist_ok=True)
        dotenv.load_dotenv()
        self.deepgram = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))

    def transcribe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transcribe all episodes in df (columns episode_hash, audio_path) in parallel.
        Returns one row per episode with episode_hash, success, path and error."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            results = list(ex.map(self.transcribe_one, df["episode_hash"], df["audio_path"]))
            return pd.DataFrame(results, columns=["episode_hash", "success", "path", "error"])

    def transcribe_one(self, episode_hash: str, episode_path: Path):
        """Transcribe one audio file, or reuse the transcript if it already exists.
        The response is written to a .part file and renamed on success, so a finished file is always complete."""
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
                    model=TRANSCRIPTION_MODEL,
                    filler_words=True, # Represents spoken language
                    diarize_model="v2", # Speaker seperation
                    paragraphs=True, # Extract speaker timings
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


def _load_alternative(path: str) -> dict:
    """Return the transcription result from a saved Deepgram JSON response."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["results"]["channels"][0]["alternatives"][0]

def load_transcript_text(path: str) -> str:
    """Return the plain transcript text (no speaker labels)."""
    return _load_alternative(path)["transcript"]

def load_transcript_words(path: str) -> list:
    """Return all words as dicts with word, punctuated_word, start, end, confidence and speaker."""
    return _load_alternative(path)["words"]

def load_transcript_sentences(path: str) -> list:
    """Return the paragraphs as dicts with sentences (text, start, end), speaker, num_words, start and end."""
    return _load_alternative(path)["paragraphs"]["paragraphs"]

def word_stats(words: list) -> dict:
    """Number of speakers, share of words spoken by the main speaker and mean word confidence."""
    if not words:
        return {"n_speakers": 0, "main_speaker_share": None, "mean_confidence": None}
    words_per_speaker = {}
    for word in words:
        words_per_speaker[word["speaker"]] = words_per_speaker.get(word["speaker"], 0) + 1
    return {
        "n_speakers": len(words_per_speaker),
        "main_speaker_share": max(words_per_speaker.values()) / len(words),
        "mean_confidence": sum(word["confidence"] for word in words) / len(words),
    }
