"""All settings that define the study, in one place.
These are the values that have to be reported and justified (see "Design Decisions" in the README).
Purely technical settings (timeouts, retries, number of workers) stay in their modules."""
from pathlib import Path

# --- Sample (Stages 1-3) ---
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
SCREENING_YEARS = [2018, 2022]  # first and last pre-period year, no post-period episode is used for screening
SEED = 23                       # goes into the hash that gives podcasts and episodes their random rank
MIN_PODCAST_LEN_MINUTES = 15    # shorter episodes are dropped

N_PODCASTS = 3                  # podcasts per group
K = 2                           # episodes per podcast and year
K_TEST = 1                      # screening episodes per podcast and screening year
BUFFER = 2                      # a podcast needs BUFFER * K episodes in every year, as reserve for failed episodes

# One entry per group: group name -> filters on the PodcastIndex table
GROUPS = {
    "first": {"language": ["en", "en-us"],
              "category1": ["games", "science"],
              "category2": ["hobbies", "medicine"],
              "category3": ["video games", "education"],
              "newestItemPubdate": [2026],
              "oldestItemPubdate": [2018],
              "dead": ["0"]},
}

# --- Download (Stage 4) ---
START_SEC, DURATION_SEC = 300, 600  # excerpt: minute 5-15

# --- Transcription (Stage 5) ---
TRANSCRIPTION_MODEL = "nova-3"

# --- Ad removal (Stage 6), not final ---
MAX_GAP_SEC = 30                # keyword hits closer than this belong to the same ad block
MIN_DISTINCT_HITS = 2           # a block needs this many different phrases to be cut
PAD_SENTENCES = 2               # sentences cut before the first and after the last hit of a block
MIN_WORDS_PER_EPISODE = 1000    # episodes with fewer words left after ad removal fail

# --- Spontaneity check (Stage 7) ---
SPONTANEITY_MODEL = "openai/gpt-oss-120b"
SCORE_THRESHOLD = 3             # scores below this count as scripted
EXCERPT_WORDS = 300             # words per episode sent to the LLM

# --- Paths ---
PROJECT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_DIR / "data" / "podcastindex_feeds.db"
DATABASE_TABLE = "podcasts"
EPISODES_PATH = PROJECT_DIR / "data" / "episodes.pkl"       # episode table (Stages 1-3)
PROCESSED_PATH = PROJECT_DIR / "data" / "processed.pkl"     # episode table with the results of Stages 4-7
DOWNLOAD_DIR = PROJECT_DIR / "downloads"
TRANSCRIPTION_DIR = PROJECT_DIR / "transcriptions" / "raw"
AD_REMOVED_TRANSCRIPTION_DIR = PROJECT_DIR / "transcriptions" / "ad_cleaned"
AD_KEYWORDS_FILE_PATH = PROJECT_DIR / "processing" / "keywords" / "ad_keywords.txt"
