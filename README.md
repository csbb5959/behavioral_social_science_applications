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

**Background.** LLM use homogenizes human language and reasoning ([Sourati et al.](https://doi.org/10.1016/j.tics.2026.01.003)). A meta-analysis finds a significant, small homogenization effect in human–AI co-creation ([de Rooij & Biskjaer](https://doi.org/10.1080/0144929X.2026.2726451)). Most of this evidence comes from written language, typically lab tasks done with or without AI.

**Why it matters.** Less diversity between people can reduce collective creativity ([Doshi & Hauser](https://doi.org/10.1126/sciadv.adn5290), [meta-analysis](https://arxiv.org/abs/2505.17241)); [Sourati et al.](https://doi.org/10.1016/j.tics.2026.01.003) give an overview.

**Gap and research question.** **Does AI also lead to homogenization in spoken language?** [Sourati et al.](https://doi.org/10.1016/j.tics.2026.01.003) suggest it but do not test it. [Yakura et al.](https://doi.org/10.48550/arXiv.2409.01754) show that AI-typical words (like "delve") increase in podcasts, but do not look at homogenization.

**Approach.** Take podcasts that ran through the whole timeframe, transcribe fixed-length excerpts per podcast and year, describe each podcast-year by its style profile, and compare how similar podcasts are to each other before and after the release of ChatGPT, in a group with high AI exposure against a control group (Difference-in-Differences). Details in the plan below.

**Open points for the introduction.**
* Literature search to confirm that homogenization of spoken language has not been studied yet (the gap is a strong claim).
* Whether to also motivate the topic via moral responsibility, in addition to collective creativity.

## Hypothesis:
- H1 (primary, tested): Podcasts after AI appearance are more similar than pre AI --> Homogenization effect
- Descriptive only, not tested: AI words increase post AI, like in the [Yakura et al.](https://doi.org/10.48550/arXiv.2409.01754) study. Our corpus is far too small to test this (see Stage 8).

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

### Design Decisions (must be made and justified before preregistration)

These choices determine what the study can show. Each one needs a short justification in the report, based on **literature** (what did comparable studies do, what are known minimum requirements) and on **logic** (why does it fit our question and budget). All of them are parameters in the code, and all of them are fixed in the preregistration. "Proposed" values come from an external review of this plan and are not final; several depend on the power simulation (see Procedure).

| # | Decision | Current state | Options / proposal | Justify with |
|---|---|---|---|---|
| 1 | **Timeframe** (which years, where pre/post is cut) | 2018–2026 in code | At least 3 pre-years. ChatGPT release: Nov 30, 2022, so 2022 counts as pre-period (pre = 2018–2022, post = 2023–2026). Decide whether 2020 is kept (see #4) | Diffusion of LLM use over time, Yakura et al., requirements of Difference-in-Differences (pre-trends) |
| 2 | **Number of podcasts per group ($N$)** | open | Result of the power simulation, limited by budget | Power simulation, effect sizes from de Rooij & Biskjaer (meta-analysis) |
| 3 | **Episodes per podcast and year ($k$) and words per episode** | $k = 2$, 1000 words | Proposed: $k$ = 3–4 and the full cleaned excerpt at a fixed length (~1400 words), i.e. roughly 5000+ words per podcast-year | Minimum text length for stable Delta values (e.g. Eder 2015, to be verified), power simulation |
| 4 | **Categories: treatment and control group** | Not final. Latest agreement: Tech (treatment), Sports (control), Politics (exploratory, out of interest). The code currently filters sports/games/science for testing only | Needs groups with clearly different AI exposure but similar format. The confirmatory test is Tech vs Sports only; Politics is reported separately. Sports and 2020 is a problem (no live sports, format changes in the middle of the pre-period), so drop or test 2020. Politics has election-year cycles. A third group adds about 50% transcription cost | Evidence on AI adoption by domain, parallel-trends assumption |
| 5 | **Primary measure** | Function words + Burrows' Delta | Function words (robust against topic, but the least sensitive part of style), function words plus discourse markers and connectives ("additionally", "overall"), or style embeddings. Content words are excluded since they confound topic | Stylometry literature (Burrows 2002), evidence on which word classes LLMs influence (Yakura et al., Juzek et al.) |
| 6 | **Size of the word list** | ~150 words | Proposed: the 50–100 most frequent, so each word occurs often enough in the available text | Stylometry literature, frequencies in our smoke-test data |
| 7 | **Excerpt** (position and length) | minute 5–15 in code, not final | Middle of the episode is probably better (less intro and fewer pre-roll ads, and seeking errors matter less), but needs the episode duration from the feed | Typical podcast structure, ad placement, cost per minute |
| 8 | **Spontaneity criterion** | LLM score 1–4 on ~300 words, threshold 3, screening on first and last year | Proposed: screen only on pre-period episodes, see Stage 7. Threshold and prompt are fixed in advance | Yakura et al. (conversation criterion), post-treatment selection bias, validation on hand-rated excerpts |
| 9 | **Ad removal rule** | keyword list + time window | Fixed before screening | Manual check of 20–30 excerpts |
| 10 | **Speakers** (all speakers vs host only) | all speakers | All speakers as primary analysis, with number of speakers and host share as control. Host-only costs text (conflicts with #3) | Function words depend on monologue vs dialogue ("I", "you") |
| 11 | **Statistical test** | Difference-in-Differences, bootstrap over podcasts | See Stage 9 for the corrected bootstrap | DiD literature, power simulation |
| 12 | **Transcription settings** | Deepgram `nova-3`, filler words kept, diarization | Fixed, report model version | Filler words are markers of spontaneous speech |

Already fixed: one pipeline for all groups with a `group` column, deterministic sampling via hash ranks, replacement rule, seed.

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
* Filter: Duration at least 15 minutes, remove missing values, year within timeframe.
* Keep a podcast only if it has at least $2 \times k$ suitable episodes in every year (buffer for replacements).

### Stage 3: Random Ranks and Status [CORE] — Completed

* `podcast_rank = Hash(Seed, feed_url)`; `rank = Hash(Seed, guid)`. Use `hashlib`, not Python’s built-in `hash()`.
* Status column for podcasts and episodes: `pending` / `passed` / `failed` + reason.
* Selection: `sort_values(["podcast_rank", "year", "rank"])`, filter out `failed` status, `groupby(["feed_url", "year"]).head(k)`.

#### Processing Order per Podcast (Important for Computational Effort)

* First, process only 2 screening episodes (Stage 4 $\to$ 5 $\to$ 6 $\to$ 7): the first in rank order from the first year (2018) and the first from the last **pre-period** year (2022). No post-period episode is used for screening (see Stage 7). Not implemented yet: the code still screens on the first and last year overall.
* Only process remaining episodes after the podcast passes the spontaneity check (Stage 7). The two screening episodes count toward this.
* If the podcast fails, only 2 audio excerpts ($20\text{ min}$ of audio) have been transcribed in vain.

### Stage 4: Download [CORE] — Completed

* For each podcast and year, take the first $k$ episodes in rank order.
* Extract excerpt from minute 5–15 using `ffmpeg` directly from the URL (`-ss` before `-i`), $16\text{ kHz}$ mono.
* Log: download date, final URL, file hash, success yes/no.
* Download failed $\to$ episode `failed`, take next in rank order.

### Stage 5: Transcription [CORE] + Speaker Seperation — Completed

* Deepgram `nova-3` via API (`filler_words=True`, `diarize_model="v2"`, `smart_format=True`). Same model and parameters for all years. The hosted model can change over time, so report the exact version stored in each response (`metadata.model_info`).
* Save the full JSON response (word-level timestamps, confidence and speaker labels). Detect language (`detect_language=True`, not enabled yet, check support for `nova-3`), non-English episodes $\to$ `failed`.
* Filler words ("uh", "um") are kept (Deepgram drops them by default), since they are a marker of spontaneous speech and matter for the spontaneity check.
* **[CHEAP]** Save mean word `confidence` per episode and report per year (transcription quality, e.g., remote recordings 2020–2021).
* **[CORE]** Save number of speakers and the share of words spoken by the main speaker per episode (from the speaker labels). Monologue and dialogue differ in function words such as "I" and "you", so this is needed as a control in Stage 9.

### Stage 6: Ad Removal

* **[CORE] — 3–4 h:** Keyword rule. Define list in advance (teammate): e.g., "sponsored by", "brought to you by", "this episode is supported by", "today's sponsor", "promo code", "use code", "discount code", "% off", "free trial", "terms apply", "go to [...].com", "visit [...].com/". Group hits closer than $120\text{ s}$ into a single block, remove from $15\text{ s}$ before to $30\text{ s}$ after. Fewer than 1000 remaining words $\to$ episode `failed`, replacement.
* Runs before the spontaneity check, so the rule must be final before screening starts.
* **[CHEAP]** Report proportion of removed time per year (indicator of dynamically inserted ads).
* **[OPTIONAL]** Calibrate rule in pilot (before screening).
* **[OPTIONAL]** Validation: Manually check 20–30 excerpts.

### Stage 7: Spontaneity Check (Runs after ad removal on the 2 screening episodes) – Completed

* **[CORE] — 4–6 h:** Podcast-level via LLM. Send ~300 words from each of the two cleaned screening transcripts to an LLM: "scripted or conversation?". If one fails $\to$ podcast `failed`, next podcast in rank order. Record prompt and model.
* Uses ad-cleaned text, since sponsor reads are scripted and would bias the rating towards "scripted". Ad removal is a cheap keyword rule and the screening episodes have to pass it anyway, so running it first costs nothing extra.
* **Screening uses pre-period episodes only.** If AI makes podcasts more scripted (hosts reading LLM-prepared text), screening on a post-period episode would remove exactly the affected podcasts and bias the result towards zero (selection on a post-treatment variable).
* **[CORE]** Compute the spontaneity score for **all** episodes, not only the screening ones (LLM cost is negligible), and report it per group and year: do podcasts become more scripted? This is a finding in its own right.
* Robustness (Stage 11): repeat the main analysis without low-score episodes, since scripted post-period episodes would measure read-out text instead of spontaneous speech.
* **[OPTIONAL]** Pre-screening with a cheaper ASR model before main transcription (saves the 2 `nova-3` transcriptions on failure).
* **[OPTIONAL]** Conversation criterion via diarization ($\geq 2$ speakers, $\geq 4$ turns, like Yakura et al.).
* **[OPTIONAL]** Yakura's classifier instead of LLM (runs directly on audio, i.e., right after Stage 4).
* **[OPTIONAL]** Validation: Read 30 excerpts and compare with LLM decision (~1 h, teammate).

#### Loop Stages 4–7 [CORE] — Included in Stage 3

* Repeat until every cell (Podcast $\times$ Year) has $k$ passed episodes and $N$ podcasts per group have passed.
* No incomplete cells (podcast-years with fewer episodes are noisier and appear more dissimilar).

### Stage 8: Measures

* **[CORE] — 4–5 h:** A fixed number of words per episode (the first ones after cleaning), identical for all episodes. Function word profile: frequencies of a pre-defined word list (teammate), z-standardized across all years and groups. Average across the $k$ episodes per podcast and year.
* **Text amount (Design Decisions #3, #6):** 2 × 1000 words per podcast-year is too little for Burrows' Delta. Rare function words then occur only one to three times, so distances are mostly noise. Since the length is the same in all years this does not bias the comparison, but it hides a small effect (low power). Proposed: full cleaned excerpt at a fixed length (~1400 words), $k$ = 3–4, word list cut to the 50–100 most frequent words. Final values come from the power simulation. The ad-removal minimum ("fewer than 1000 remaining words", Stage 6) must be raised to the chosen length.
* **[CHEAP]** Complexity metrics according to Sourati et al. (2026): Type-Token Ratio, Shannon Entropy, Simpson Index, Hapax Legomena on the same words.
* **[CHEAP]** Rate of known ChatGPT words per year (list by Juzek et al., `[github.com/fsu-nlp/ai-traces-ssl](https://github.com/fsu-nlp/ai-traces-ssl)`). **Descriptive only, no hypothesis test:** these words occur a few times per million words, so with our corpus size the expected counts per group and year are far too small to detect an increase (Yakura et al. used a corpus several orders of magnitude larger).
* **[OPTIONAL]** Style Embeddings (e.g., CISR).
* **[OPTIONAL]** Semantic embeddings as a topic control.

### Stage 9: Statistics [CORE] — 6–8 h

* Metric: Mean pairwise Burrows' Delta distance between podcasts per group and year. Homogenization = distance decreases.
* Test: $(\text{Post-Mean} - \text{Pre-Mean})$ in Tech minus the same in Sports (Difference-in-Differences).
* Confidence interval via bootstrap over podcasts (not episodes). **Avoid self-pairs:** resampling podcasts with replacement and then computing pairwise distances creates duplicate pairs with distance 0, which pulls the mean distance down and distorts the interval. Instead compute, once on the original data, each podcast's mean distance to all other podcasts of its group per year (one value per podcast × year), then bootstrap podcasts over these values.
* Control for speaker composition: number of speakers and main-speaker share (Stage 5) as covariate or robustness check.
* Plot: Trajectory of both groups over the years, visually compare pre-trends. If Sports is the control group, check 2020 explicitly (Design Decisions #4).
* **[CORE] Power simulation before preregistration — 3–4 h:** take the smoke-test/pilot profiles, build synthetic datasets with a small built-in homogenization effect, and count how often the test detects it for different $N$, $k$, text lengths and word-list sizes. This decides Design Decisions #2, #3 and #6, and whether the full run is worth its cost.
* **[OPTIONAL]** Permutation test (permute year labels within each podcast).
* **[CHEAP]** Same analysis with Sourati metrics (variance between podcasts per year).

### Stage 10: Documentation [CORE]

* **[CORE] — 4–6 h:** Flowchart of exclusions, limitations, and discussion of potential biases. Include the exact prompt and model used for the spontaneity check.
* **[CORE]** tbd

### Stage 11: Robustness [OPTIONAL]

* Different excerpt (minute 20–30), without ad cleaning, host-only (main speaker's words only, leaves less text), style embeddings.
* Without low spontaneity-score episodes (Stage 7), without 2020 if Sports is the control group.

---

### Procedure

1. Code Stages 3–7 [CORE]
2. Smoke Test [CORE] — 8–10 h: 1 podcast per group run completely through the pipeline, for debugging only, not for testing the hypothesis.
* **[OPTIONAL]** Pilot with 3–5 podcasts for calibration and estimation of $N$.
3. Power simulation [CORE] with the smoke-test/pilot data (Stage 9). Then settle all Design Decisions and write down the justification for each.
4. Preregistration [CORE, Teammates]: Brief, 1–2 pages, OSF template. Hypothesis, all Design Decisions (timeframe, groups, $N$, $k$, text length, primary measure, word list), test, exclusion rules, replacement rule, seed, keyword list.
5. Full Run [CORE]: Deepgram cost scales with audio minutes: groups × $N$ × years × $k$ × 10 min (e.g. 2 × 30 × 9 × 4 × 10 = 21,600 min). Check the current price per minute and the remaining credit before starting.
6. Analysis Stages 8–9 [CORE]: Can be prepared by a teammate using smoke test data.
7. Report [CORE, Team]: Including flowchart of exclusions and limitations.

---

### Work Distribution

Lucas: Stages 3–7
Carma: Stages 8–11
Gaia: Writing, report

---

### Limitations for the Report (One sentence each on why it matters)

1. **Survivorship bias via panel design:** Only podcasts that survived the entire timeframe.
2. **Host and guests analysed together;** speaker composition is only controlled via number of speakers and main-speaker share.
3. **Spontaneity check and ad rule not manually validated.**
4. **ASR smooths style** (`smart_format` normalizes punctuation and numbers, filler words are kept but may not be captured completely) $\to$ effect likely underestimated.
5. **Dynamically inserted ads removed only via keywords;** residuals possible.
6. **Control group also has AI exposure, just less** $\to$ effect likely underestimated.
7. **Parallel pre-trends only roughly testable** (few pre-years).
8. **English only, few categories.**
9. **Limited text per podcast and year** $\to$ noisy distances, low power for a small effect; a null result is not evidence of no effect.
10. **Function words are the most stable part of style,** while the documented LLM influence is on content words $\to$ effect likely underestimated.
11. **AI-word rates are descriptive only** (corpus too small for a test).
12. **Excerpt position is approximate:** `ffmpeg` seeking in variable-bitrate MP3s is estimated, and dynamically inserted ads shift the content. Random, not systematic.
13. **Spontaneity is screened in the pre-period only;** podcasts may become more scripted later (reported via the per-episode score).



## Sources
- Anderson, B. R., Shah, J. H., & Kreminski, M. (2024). Homogenization Effects of Large Language Models on Human Creative Ideation. Proceedings of the 16th Conference on Creativity & Cognition, 413–425. https://doi.org/10.1145/3635636.3656204
- Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜. Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’21, 610–623. https://doi.org/10.1145/3442188.3445922
- Bommasani, R., Creel, K. A., Kumar, A., Jurafsky, D., & Liang, P. (2022). Picking on the Same Person: Does Algorithmic Monoculture Lead to Outcome Homogenization? Advances in Neural Information Processing Systems, 35. https://doi.org/10.52202/068431-0265
- de Rooij, A., & Biskjaer, M. M. (2026). Does generative AI make us think alike? A systematic review and meta-analysis of homogenisation effects in human–AI co-creation. Behaviour & Information Technology, 0(0), 1–14. https://doi.org/10.1080/0144929X.2026.2726451
- Doshi, A. R., & Hauser, O. P. (2024). Generative AI enhances individual creativity but reduces the collective diversity of novel content. Science Advances, 10(28), eadn5290. https://doi.org/10.1126/sciadv.adn5290
- Duede, E., Teplitskiy, M., Lakhani, K., & Evans, J. (2024). Being together in place as a catalyst for scientific advance. Research Policy, 53(2), 104911. https://doi.org/10.1016/j.respol.2023.104911
- Hong, L., & Page, S. E. (2004). Groups of diverse problem solvers can outperform groups of high-ability problem solvers. Proceedings of the National Academy of Sciences, 101(46), 16385–16389. https://doi.org/10.1073/pnas.0403723101
- Muthukrishna, M., & Henrich, J. (2016). Innovation in the collective brain. Philosophical Transactions of the Royal Society B: Biological Sciences, 371(1690), 20150192. https://doi.org/10.1098/rstb.2015.0192
- Nijstad, B. A., Stroebe, W., & Lodewijkx, H. F. M. (2002). Cognitive stimulation and interference in groups: Exposure effects in an idea generation task. Journal of Experimental Social Psychology, 38(6), 535–544. https://doi.org/10.1016/S0022-1031(02)00500-0
- Page, S. E. (2008). The Difference: How the Power of Diversity Creates Better Groups, Firms, Schools, and Societies. Princeton University Press.
- Sourati, Z., S. Ziabari, A., & Dehghani, M. (2026). The homogenizing effect of large language models on human expression and thought. Trends in Cognitive Sciences, 30(9), 805–816. https://doi.org/10.1016/j.tics.2026.01.003
- Yakura, H., Lopez-Lopez, E., Brinkmann, L., de la Serna, I., Kirfel, L., Gupta, P., Soraperra, I., Eisenmann, T. F., Wulff, D. U., & Rahwan, I. (2024). Empirical evidence of Large Language Model's influence on human spoken communication. arXiv. https://doi.org/10.48550/arXiv.2409.01754

To do: one entry could not be identified (Scopus record https://www.scopus.com/pages/publications/85166037135). Sources used in the plan but still missing here: Burrows (2002), Eder (2015), Juzek et al.
