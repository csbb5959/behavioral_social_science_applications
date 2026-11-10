from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import logging
import os, json
from groq import Groq
import dotenv
from transcribe import load_transcript_words

MODEL = "openai/gpt-oss-120b"
PROMPT = """You get an excerpt from a podcast transcript.
Each line is one speaker turn, labeled by automatic speaker diarization.
Rate how spontaneous the speech is:
1 = clearly scripted (read from a prepared text, like news or an audiobook)
2 = mostly scripted (prepared text with a few free parts)
3 = mostly spontaneous (free talk with a few prepared parts)
4 = clearly spontaneous (free conversation: hesitations, interruptions, self-corrections)
First give a one-sentence reason, then the score.
Answer only with JSON: {"reason": "...", "score": <1-4>}"""
SCORE_THRESHOLD = 3     # scores below this count as scripted
EXCERPT_WORDS = 300     # words per episode sent to the LLM for classification

class SpontaneityClassifier():
    def __init__(self, max_workers: int = 4, temperature: int=0):
        self.max_workers = max_workers
        self.temperature = temperature
        dotenv.load_dotenv()
        self.model = Groq(api_key=os.environ["GROQ_API_KEY"])

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            results = list(ex.map(self.__classify_one, df["episode_hash"], df["transcript_path"]))
        return pd.DataFrame(results, columns=["episode_hash", "success", "score", "scripted", "scripted_reason", "model", "error"])

    def __classify_one(self, episode_hash: str, transcript_path: str):
        try:
            logging.info(f"Starting spontaneity classification for {episode_hash}")
            excerpt = build_excerpt(load_transcript_words(transcript_path), EXCERPT_WORDS)

            response = self.model.chat.completions.create(
                model=MODEL,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": PROMPT},
                          {"role": "user", "content": excerpt}],
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from LLM")
            message = json.loads(content)
            score = int(message["score"])
            if score not in (1, 2, 3, 4):
                raise ValueError(f"Score out of range: {score}")
            scripted = score < SCORE_THRESHOLD

            return episode_hash, True, score, scripted, message["reason"], response.model, None

        except Exception as e:
            logging.error(f"Error during spontaneity check for {episode_hash}: {e}")
            return episode_hash, False, None, None, None, None, repr(e)


def build_excerpt(words: list, n_words: int) -> str:
    """First n_words words, one line per speaker turn, e.g. "Speaker 0: Hi there."
    Labels are added on top, so exactly n_words spoken words are sent."""
    if words and "speaker" not in words[0]:
        raise ValueError("Transcript has no speaker labels, re-transcribe with diarization")

    turns = []                                  # list of (speaker, [words])
    for w in words[:n_words]:
        if not turns or turns[-1][0] != w["speaker"]: # if loaded first or the last speaker is not the current speaker
            turns.append((w["speaker"], [])) # append the new speaker
        turns[-1][1].append(w["punctuated_word"]) # append the word of the speaker
    return "\n".join(f"Speaker {speaker}: {' '.join(turn_words)}" for speaker, turn_words in turns)
