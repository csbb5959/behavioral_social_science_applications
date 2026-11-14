import logging
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from config import START_SEC, DURATION_SEC

TIMEOUT_SEC = 300
MAX_ATTEMPTS = 3
RETRY_WAIT_SEC = 5                      
TRANSIENT_ERRORS = ("timeout", "timed out", "connection reset", "broken pipe",
                    "server returned 5", "temporary failure in name resolution")

class EpisodeDownloader:
    """Downloads a fixed excerpt of each episode as 16 kHz mono wav file with ffmpeg."""
    def __init__(self, out_dir: Path, max_workers: int = 6):
        """Create the output folder. max_workers is the number of parallel downloads."""
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

    def download(self, df: pd.DataFrame) -> pd.DataFrame:
        """Download all episodes in df (columns episode_hash, episode_audio_url) in parallel.
        Returns one row per episode with episode_hash, success, path and error."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            results = list(ex.map(self.__download_one, df["episode_hash"], df["episode_audio_url"]))
        return pd.DataFrame(results, columns=["episode_hash", "success", "path", "error"])

    def __download_one(self, episode_hash: str, url: str):
        """Download one episode, or reuse the file if it already exists.
        Transient errors (e.g. timeouts) are retried up to MAX_ATTEMPTS times with a growing wait."""
        out = self.out_dir / f"{episode_hash}.wav"
        if out.exists():
            return episode_hash, True, str(out), None

        err = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            err = self.__run_ffmpeg(url, out)
            if err is None:
                logging.info(f"Download OK: {episode_hash}")
                return episode_hash, True, str(out), None
            if attempt == MAX_ATTEMPTS or not self.__is_transient(err):
                break
            logging.warning(f"RETRY {episode_hash} ({attempt}/{MAX_ATTEMPTS}): {err}")
            time.sleep(RETRY_WAIT_SEC * attempt)

        logging.error(f"FAIL {episode_hash}: {err}")
        return episode_hash, False, None, err

    def __run_ffmpeg(self, url: str, out: Path) -> str | None:
        """Run ffmpeg once. Writes to a .part file and renames it on success, so a finished file is always complete.
        Returns None on success, otherwise the error text."""
        tmp = out.with_suffix(".part.wav")
        cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
               "-ss", str(START_SEC), "-t", str(DURATION_SEC), "-i", url,
               "-ar", "16000", "-ac", "1", str(tmp)]
        try:
            subprocess.run(cmd, check=True, timeout=TIMEOUT_SEC, capture_output=True, text=True)
            tmp.rename(out)
            return None

        except subprocess.TimeoutExpired:
            err = "timeout"

        except subprocess.CalledProcessError as e:
            err = (e.stderr or "").strip()[-300:] or f"ffmpeg exit {e.returncode}"

        except FileNotFoundError as e:
            err = e.filename

        except Exception as e:
            err = repr(e)

        tmp.unlink(missing_ok=True)
        return err

    @staticmethod
    def __is_transient(err: str) -> bool:
        """Return True if the error text looks temporary, so that a retry makes sense."""
        err = err.lower()
        return any(t in err for t in TRANSIENT_ERRORS)