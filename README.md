# Behavioral Social Sciences Applications Project

## Quick Start

**Requirements:** conda ([Miniforge](https://github.com/conda-forge/miniforge) or [Anaconda](https://www.anaconda.com/download)), a [Deepgram](https://deepgram.com) API key, and the [PodcastIndex](https://podcastindex.org) database dump (~5 GB).

```bash
conda env create -f environment.yml    # Python 3.13, ffmpeg, all packages
conda activate bssa_project

cp .env.example .env                   # add your DEEPGRAM_API_KEY
```

With Anaconda Navigator, you can instead import `environment.yml` under *Environments → Import*.

Place the extracted dump at `data/podcastindex_feeds.db`.

**Run:** open `processing/test.ipynb`, select the `bssa_project` kernel, and run all cells. This filters candidate podcasts, builds the ranked episode table, then downloads and transcribes one test episode (audio → `downloads/`, transcripts → `transcriptions/`).

Notebooks must run with their own folder as working directory (VS Code default), since paths like `../data/` are relative. The full screening loop (`processing/processing_pipeline.py`) is work in progress.

----------------------------

## General Idea

[Sourati et al.](https://doi.org/10.1016/j.tics.2026.01.003) (2nd paper from AI Ethics) show: 
- Homogenization happens on the language and reasoning level etc. --> Meta analysis showed significant, small effect [de Rooij et al.](https://doi.org/10.1080/0144929X.2026.2726451)
- Most studies inspect written language, specifically lab tests where subjects to exercises mostly with or without AI

[Doshi](https://scholar.google.com/scholar_lookup?title=Generative%20AI%20enhances%20individual%20creativity%20but%20reduces%20the%20collective%20diversity%20of%20novel%20content&author=A.R.%20Doshi&publication_year=2024) and many other studies ([also meta analysis](https://arxiv.org/abs/2505.17241)) show that this can have negative effects on collective creativity for example (can I add "and moral responsibility?" so I can motivate even more why we chose this topic and I really like writing about it hihi). [Sourati et al.](https://doi.org/10.1016/j.tics.2026.01.003) give a good overview of this. So studying this is important cause it could have significant effects on global creativity (and morality! You will see tomorrow).

What has not been done/studied thoroughly before: 

--> **Does AI also lead to homogenization in spoken language?** [Sourati et al.](https://doi.org/10.1016/j.tics.2026.01.003) hint this, but don't inspect it. [Yakura et al.](https://doi.org/10.48550/arXiv.2409.01754) show that AI words increase in podcasts (like "dvelve"), but dont inspect homogenization. 
(Are we sure there are no studies on this? Cause this is a strong claim).

## Process

1. Find ~20-100 podcasts pre and post AI with specific time windows (what about political discourses? too difficult?) (and are we able to find podasts from before 2021?)
2. Use 10-30 minutes of podcasts per time window and transcribe them
3. Analyse functional words / maybe embeddings as well & extract AI keywords like [Yakura et al.](https://doi.org/10.48550/arXiv.2409.01754) (2nd one is important to see if the same AI effect happens in our sample, if not, homogenization is less likely as well)
4. Compare functional words between podcasts pre and post AI with **Difference in Differences** (DiD)

## Hypothesis:
- H1: AI words increase post AI, just like in [Yakura et al.](https://doi.org/10.48550/arXiv.2409.01754) study
- H2: Podcasts after AI appearance are more similar then pre AI --> Homogenization effect

----------------------------

## PLAN – Homogenization of Spoken Language in Podcasts
Course project, 2.5 ECTS (~60–75 h per person), 3 members

---

### Labels

* **[CORE]** Necessary for a valid course project
* **[CHEAP]** Optional, 1–2 h, uses data generated anyway — do if time permits
* **[OPTIONAL]** Publication level — omit and mention in limitations
* Hours = rough estimate for coding share (Lucas), unless otherwise stated.

---

### Fixed Parameters

* **Control group:** Sports (Tech vs Sports vs Politics). 
Three categories with different levels of AI exposure. A "group" column in all tables, a single pipeline for both groups.
* **Primary measure:** Function word profiles with Burrows’ Delta.
* **Sample:** N = unclear podcasts per group, k = 2 episodes per year (fixed, not estimated via a pilot).

---

### Still Open

* **Timeframe:** 2018–2026. Treated as a parameter in the code. ChatGPT was released on November 30, 2022; at least three pre-years are sensible.

---

### Basic Principle

Each stage writes a file and skips already processed entries. No random sampling is done on the fly: every podcast and episode receives a fixed random rank once (hash of seed and ID). The first elements in rank order that pass all checks are selected. If one fails, the next in line takes its place. Every exclusion is logged with a reason (flowchart in the report).

---

### Stage 1: Candidates [CORE] — Completed

* Filter PodcastIndex dump: language, category, oldest episode before timeframe start, newest episode 2026, not dead.
* Note the snapshot date of the dump.
* **[OPTIONAL]** Intersection with Yakura's dataset of podcasts classified as spontaneous (request access via Zenodo).

### Stage 2: Episode Table [CORE] — Completed

* Parse feeds. Columns: `feed_url`, `group`, `guid`, `year`, `pub_date`, `title`, `duration`, `audio_url`.
* Filter: Duration 15 minutes, remove missing values, year within timeframe.
* Keep a podcast only if it has at least 2k suitable episodes in every year (buffer for replacements).

### Stage 3: Random Ranks and Status [CORE] — 2–3 h

* `podcast_rank = Hash(Seed, feed_url)`; `rank = Hash(Seed, guid)`. Use `hashlib`, not Python’s built-in `hash()`.
* Status column for podcasts and episodes: `pending` / `passed` / `failed` + reason.
* Selection: `sort_values(["podcast_rank", "year", "rank"])`, filter out `failed` status, `groupby(["feed_url", "year"]).head(k)`.

#### Processing Order per Podcast (Important for Computational Effort)

* First, process only 2 screening episodes (Stage 4 \to 5 \to 6): the first in rank order from the first year and the first from the last year. This also flags format changes over time.
* Only process remaining episodes after the podcast passes the spontaneity check (Stage 6). The two screening episodes count toward this.
* If the podcast fails, only 2 audio excerpts ($20\text{ min}$ of audio) have been transcribed in vain.

### Stage 4: Download [CORE] — 4–6 h

* For each podcast and year, take the first $k$ episodes in rank order.
* Extract excerpt from minute 5–15 using `ffmpeg` directly from the URL (`-ss` before `-i`), $16\text{ kHz}$ mono.
* Log: download date, final URL, file hash, success yes/no.
* Download failed $\to$ episode `failed`, take next in rank order.

### Stage 5: Transcription [CORE] — 4–6 h

* `faster-whisper large-v3` on Kaggle (GPU). Fixed version and parameters for all years: `vad_filter=True`, `condition_on_previous_text=False`, `word_timestamps=True`.
* Save segments with timestamps. Save detected language, non-English episodes $\to$ `failed`.
* **[CHEAP]** Save `avg_logprob` per episode and report per year (transcription quality, e.g., remote recordings 2020–2021).

### Stage 6: Spontaneity Check (Runs after transcribing the 2 screening episodes)

* **[CORE] — 4–6 h:** Podcast-level via LLM. Send ~300 words from each of the two screening transcripts to an LLM: "scripted or conversation?". If one fails $\to$ podcast `failed`, next podcast in rank order. Record prompt and model.
* **[OPTIONAL]** Pre-screening with a small custom Whisper model before main transcription (saves the 2 `large-v3` transcriptions on failure).
* **[OPTIONAL]** Conversation criterion via diarization ($\geq 2$ speakers, $\geq 4$ turns, like Yakura et al.).
* **[OPTIONAL]** Yakura's classifier instead of LLM (runs directly on audio, i.e., right after Stage 4).
* **[OPTIONAL]** Validation: Read 30 excerpts and compare with LLM decision (~1 h, teammate).

### Stage 7: Ad Removal

* **[CORE] — 3–4 h:** Keyword rule. Define list in advance (teammate): e.g., "sponsored by", "brought to you by", "this episode is supported by", "today's sponsor", "promo code", "use code", "discount code", "% off", "free trial", "terms apply", "go to [...].com", "visit [...].com/". Group hits closer than $120\text{ s}$ into a single block, remove from $15\text{ s}$ before to $30\text{ s}$ after. Fewer than 1000 remaining words $\to$ episode `failed`, replacement.
* **[CHEAP]** Report proportion of removed time per year (indicator of dynamically inserted ads).
* **[OPTIONAL]** Calibrate rule in pilot.
* **[OPTIONAL]** Validation: Manually check 20–30 excerpts.

#### Loop Stages 4–7 [CORE] — Included in Stage 3

* Repeat until every cell (Podcast $\times$ Year) has $k$ passed episodes and $N$ podcasts per group have passed.
* No incomplete cells (podcast-years with fewer episodes are noisier and appear more dissimilar).

### Stage 8: Speaker Separation [OPTIONAL]

* `pyannote speaker-diarization-3.1`, host = most speaking time, use only host words.
* Without: Limitation (shifting guests can influence similarity).

### Stage 9: Measures

* **[CORE] — 4–5 h:** Exactly 1000 words per episode (the first ones after cleaning). Function word profile: frequencies of ~150 function words and discourse markers (pre-defined list, teammate), z-standardized across all years and groups. Average across the $k$ episodes per podcast and year.
* **[CHEAP]** Complexity metrics according to Sourati et al. (2026): Type-Token Ratio, Shannon Entropy, Simpson Index, Hapax Legomena on the same 1000 words.
* **[CHEAP]** Rate of known ChatGPT words per year (list by Juzek et al., `[github.com/fsu-nlp/ai-traces-ssl](https://github.com/fsu-nlp/ai-traces-ssl)`) as a sanity check.
* **[OPTIONAL]** Style Embeddings (e.g., CISR).
* **[OPTIONAL]** Semantic embeddings as a topic control.

### Stage 10: Statistics [CORE] — 6–8 h

* Metric: Mean pairwise Burrows' Delta distance between podcasts per group and year. Homogenization = distance decreases.
* Test: $(\text{Post-Mean} - \text{Pre-Mean})$ in Tech minus the same in Sports (Difference-in-Differences).
* Confidence interval via bootstrap over podcasts (not episodes).
* Plot: Trajectory of both groups over the years, visually compare pre-trends.
* **[OPTIONAL]** Permutation test (permute year labels within each podcast).
* **[CHEAP]** Same analysis with Sourati metrics (variance between podcasts per year).

### Stage 11: Robustness [OPTIONAL]

* Different excerpt (minute 20–30), without ad cleaning, host-only (if Stage 8 omitted), style embeddings.

---

### Procedure

1. Code Stages 3–7 [CORE]
2. Smoke Test [CORE] — 8–10 h: 1 podcast per group run completely through the pipeline, for debugging only, not for testing the hypothesis.
* **[OPTIONAL]** Pilot with 3–5 podcasts for calibration and estimation of $N$.


3. Preregistration [CORE, Teammates]: Brief, 1–2 pages, OSF template. Hypothesis, timeframe, groups, primary measure, test, exclusion rules, replacement rule, seed, keyword list, function word list.
4. Full Run [CORE]: Roughly 10–20 GPU hours (Kaggle ~30 h/week), 1–2 weeks calendar time.
5. Analysis Stages 9–10 [CORE]: Can be prepared by a teammate using smoke test data.
6. Report [CORE, Team]: Including flowchart of exclusions and limitations.

---

### Work Distribution

unclar

---

### Limitations for the Report (One sentence each on why it matters)

1. **Survivorship bias via panel design:** Only podcasts that survived the entire timeframe.
2. **No separation of host and guest** (if Stage 8 omitted).
3. **Spontaneity check and ad rule not manually validated.**
4. **Whisper smooths style (filler words)** $\to$ effect likely underestimated.
5. **Dynamically inserted ads removed only via keywords;** residuals possible.
6. **Control group also has AI exposure, just less** $\to$ effect likely underestimated.
7. **Parallel pre-trends only roughly testable** (few pre-years).
8. **English only, only two categories.**

---

*Calibrated probability:* ~70% that the core version is feasible to this extent. ~65% that function word profiles provide enough signal for a small effect (speculation); a null result is possible and, with clean preregistration, still a valid outcome.



## Sources
- Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜. Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’21, 610–623. https://doi.org/10.1145/3442188.3445922
- Doshi: Generative AI enhances individual creativity... - Google Scholar. (o. J.). Abgerufen 27. Oktober 2026, von https://scholar.google.com/scholar_lookup?title=Generative%20AI%20enhances%20individual%20creativity%20but%20reduces%20the%20collective%20diversity%20of%20novel%20content&author=A.R.%20Doshi&publication_year=2024
- Duede, E., Teplitskiy, M., Lakhani, K., & Evans, J. (2024). Being together in place as a catalyst for scientific advance. Research Policy, 53(2), 104911. https://doi.org/10.1016/j.respol.2023.104911
- Homogenization Effects of Large Language Models on Human Creative Ideation | Proceedings of the 16th Conference on Creativity & Cognition. (o. J.). Abgerufen 27. Oktober 2026, von https://dl.acm.org/doi/10.1145/3635636.3656204
- Hong, L., & Page, S. E. (2004). Groups of diverse problem solvers can outperform groups of high-ability problem solvers. Proceedings of the National Academy of Sciences, 101(46), 16385–16389. https://doi.org/10.1073/pnas.0403723101
- Muthukrishna, M., & Henrich, J. (2016). Innovation in the collective brain. Philosophical Transactions of the Royal Society B: Biological Sciences, 371(1690), 20150192. https://doi.org/10.1098/rstb.2015.0192
- Nijstad, B. A., Stroebe, W., & Lodewijkx, H. F. M. (2002). Cognitive stimulation and interference in groups: Exposure effects in an idea generation task. Journal of Experimental Social Psychology, 38(6), 535–544. https://doi.org/10.1016/S0022-1031(02)00500-0
- Page: The difference: How the power of diversity... - Google Scholar. (o. J.). Abgerufen 27. Oktober 2026, von https://scholar.google.com/scholar_lookup?title=The%20Difference%3A%20How%20the%20Power%20of%20Diversity%20Creates%20Better%20Groups%2C%20Firms%2C%20Schools%2C%20and%20Societies&author=S.E.%20Page&publication_year=2008
- Picking on the Same Person: Does Algorithmic Monoculture Lead to Outcome Homogenization? (o. J.). Proceedings.Com. Abgerufen 27. Oktober 2026, von https://www.proceedings.com/068431-0265.html
- Scopus—Document Details. (o. J.). Abgerufen 27. Oktober 2026, von https://www.scopus.com/pages/publications/85166037135
- Sourati, Z., S. Ziabari, A., & Dehghani, M. (2026). The homogenizing effect of large language models on human expression and thought. Trends in Cognitive Sciences, 30(9), 805–816. https://doi.org/10.1016/j.tics.2026.01.003
- de Rooij, A., & Biskjaer, M. M. (2026). Does generative AI make us think alike? A systematic review and meta-analysis of homogenisation effects in human–AI co-creation. Behaviour & Information Technology, 0(0), 1–14. https://doi.org/10.1080/0144929X.2026.2726451
- (PDF) Empirical evidence of Large Language Model’s influence on human spoken communication. (2026, Juli 17). ResearchGate. https://doi.org/10.48550/arXiv.2409.01754

