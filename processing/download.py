import logging
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

DOWNLOAD_DIR = Path(__file__).resolve().parent.parent / "downloads"
START_SEC, DURATION_SEC = 300, 600      # minute 5-15
TIMEOUT_SEC = 300
MAX_ATTEMPTS = 3
RETRY_WAIT_SEC = 5                      
TRANSIENT_ERRORS = ("timeout", "timed out", "connection reset", "broken pipe",
                    "server returned 5", "temporary failure in name resolution")

class EpisodeDownloader:
    def __init__(self, out_dir: Path = DOWNLOAD_DIR, max_workers: int = 6):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

    def download(self, df: pd.DataFrame) -> pd.DataFrame:
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            results = list(ex.map(self.__download_one, df["episode_hash"], df["episode_audio_url"]))
        return pd.DataFrame(results, columns=["episode_hash", "success", "path", "error"])

    def __download_one(self, episode_hash: str, url: str):
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
        err = err.lower()
        return any(t in err for t in TRANSIENT_ERRORS)