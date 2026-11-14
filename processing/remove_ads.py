from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import json
import logging
import re
from transcribe import load_transcript_sentences, load_transcript_words, word_stats
from config import AD_KEYWORDS_FILE_PATH, MAX_GAP_SEC, MIN_DISTINCT_HITS, PAD_SENTENCES

class AdRemover():
    """Marks ad sentences in a transcript with a keyword rule and saves the result as a second JSON file.
    The raw transcript is never changed."""
    def __init__(self, out_dir: Path, max_workers: int=4):
        """Create the output folder, load the keyword list and build one regex pattern from all phrases."""
        self.out_dir = out_dir
        self.max_workers = max_workers
        self.out_dir.mkdir(parents=True, exist_ok=True)
        keywords = self.__get_ad_keyword_list(keyword_file_path=AD_KEYWORDS_FILE_PATH)
        keywords.sort(key=len, reverse=True) # longest first, so "limited time offer" wins over shorter phrases
        self.pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in keywords) + r")(?!\w)") # One pattern for all phrases, matched as whole phrases (not inside longer words)

    def remove_podcasts_ads(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mark ads in all episodes in df (columns episode_hash, transcript_path).
        Returns one row per episode with episode_hash, success, path, n_words (words left without ads), n_speakers,
        main_speaker_share, mean_confidence and error."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            results = list(ex.map(self.__remove_one_podcast_ad, df["episode_hash"], df["transcript_path"]))
        return pd.DataFrame(results, columns=["episode_hash", "success", "path", "n_words", "n_speakers",
                                              "main_speaker_share", "mean_confidence", "error"])

    def __remove_one_podcast_ad(self, episode_hash: str, transcript_path: str):
        """Mark the ads of one episode, or reuse the file if it already exists.
        Sentences with keyword hits that are close in time form a block. A block with at least MIN_DISTINCT_HITS different
        phrases is marked as ad, together with PAD_SENTENCES sentences before and after it."""
        out = self.out_dir / f"{episode_hash}.json"
        tmp = out.with_suffix(".part.json")

        try:
            if out.exists():
                return self.__result(episode_hash, out)

            paragraphs = load_transcript_sentences(path=transcript_path)
            sentences = [sentence for paragraph in paragraphs for sentence in paragraph["sentences"]] # Flat list pointing to the same sentence dicts, so changes also show up in paragraphs

            # 1. Mark keyword hits per sentence
            for sentence in sentences:
                sentence["hits"] = self.pattern.findall(normalize(sentence["text"]))
                sentence["is_ad"] = False

            # 2. Group sentences with hits into blocks and cut blocks with enough different phrases
            for block in find_blocks(sentences):
                phrases = set()
                for i in block:
                    phrases.update(sentences[i]["hits"])
                if len(phrases) < MIN_DISTINCT_HITS:
                    continue
                else:
                    first = max(block[0] - PAD_SENTENCES, 0) # make sure index cannot be negative with max
                    last = min(block[-1] + PAD_SENTENCES, len(sentences) - 1) # make sure index cannot be negative higher than last sentence with min
                    for i in range(first, last + 1):
                        sentences[i]["is_ad"] = True

            # 3. Save in the same structure as the transcript sentences, plus hits and is_ad.
            # Word counts and stats use the Deepgram word list, the same words the measures use later.
            words = load_transcript_words(transcript_path)
            kept_words = keep_words(words, sentences)
            result = {
                "n_words_total": len(words),
                "n_words_kept": len(kept_words),
                **word_stats(kept_words), # n_speakers, main_speaker_share, mean_confidence
                "paragraphs": paragraphs,
            }
            
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
            tmp.rename(out)
            return self.__result(episode_hash, out)

        except Exception as e:
            logging.error(f"Error during ad removal for {episode_hash}: {e}")
            tmp.unlink(missing_ok=True)
            return episode_hash, False, None, None, None, None, None, repr(e)

    def __result(self, episode_hash: str, out: Path):
        """Result row for a finished episode, read from the saved file (also used when the file already existed)."""
        with open(out, encoding="utf-8") as f:
            cleaned = json.load(f)
        return (episode_hash, True, str(out), cleaned["n_words_kept"], cleaned["n_speakers"],
                cleaned["main_speaker_share"], cleaned["mean_confidence"], None)

    def __get_ad_keyword_list(self, keyword_file_path: Path) -> list:
        """Read the phrases from the keyword file. Empty lines and lines starting with "#" are skipped."""
        ad_keyword_lines = []
        with open(file=keyword_file_path, mode="r", encoding="utf-8") as keywords_f:
            for line in keywords_f.readlines():
                if line.startswith("#") or line.strip() == "":
                    continue
                else:
                    ad_keyword_lines.append(line.strip())
        return ad_keyword_lines

def normalize(text: str) -> str:
    """Lowercase, hyphens to spaces, drop punctuation except apostrophes and "%" (same form as the keyword file)."""
    text = text.lower().replace("-", " ").replace("’", "'")
    text = text.replace("%", " %")  # "20% off" -> "20 % off", so the phrase "% off" can match
    text = re.sub(r"[^\w\s'%]", "", text)
    return " ".join(text.split())

def find_blocks(sentences: list) -> list:
    """Return lists of sentence indices. Each list holds sentences with hits that are at most MAX_GAP_SEC apart."""
    blocks = []
    last_hit_end = None                 # end time of the previous sentence with a hit
    for i, sentence in enumerate(sentences):
        if not sentence["hits"]:
            continue
        is_close = last_hit_end is not None and sentence["start"] - last_hit_end <= MAX_GAP_SEC
        if is_close:
            blocks[-1].append(i)        # same block as the previous hit
        else:
            blocks.append([i])          # start a new block
        last_hit_end = sentence["end"]
    return blocks

def keep_words(words: list, sentences: list) -> list:
    """Return the words that are not inside a sentence marked as ad (matched by start time)."""
    ad_spans = [(s["start"], s["end"]) for s in sentences if s["is_ad"]]
    kept = []
    for word in words:
        in_ad = False
        for start, end in ad_spans:
            if start <= word["start"] < end:
                in_ad = True
                break
        if not in_ad:
            kept.append(word)
    return kept

def load_cleaned_words(transcript_path: str, cleaned_path: str) -> list:
    """Return the words of the raw transcript without ads (dicts like load_transcript_words)."""
    with open(cleaned_path, encoding="utf-8") as f:
        paragraphs = json.load(f)["paragraphs"]
    sentences = [sentence for paragraph in paragraphs for sentence in paragraph["sentences"]]
    return keep_words(load_transcript_words(transcript_path), sentences)
