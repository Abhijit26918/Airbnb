# PriceLens — Airbnb Listing Price Intelligence
## Project Plan & Execution Steps

> **Read this file first. Then read `TECHNICAL_DESIGN.md`.**
> This document is the *what* and *when*. The design doc is the *how*.
> Both are written to be consumed by Claude Code in VS Code as working context.

---

## 0. The one-line pitch

> A host in {CITY} pastes their listing details and gets back a **defensible nightly price band** (not a single number), the **top drivers** of that price, and the **five most comparable listings** — built on 75-column real-world scraped data with free text, amenity lists, and geography.

**Why this beats the average Airbnb-price notebook:** almost every version of this project on GitHub does `RandomForest(price ~ tabular)` on `listings.csv`, reports R² ≈ 0.55, and stops. Ours differs on six axes, each of which is a talking point in an interview or a report section:

| # | Differentiator | Why it matters |
|---|---|---|
| 1 | **Prediction intervals, not point estimates** (quantile GBM + split-conformal calibration) | A host cannot act on "€142". They can act on "€128–€165, and you're currently at €99 → underpriced". |
| 2 | **Cold-start vs. warm model split** | Review-derived features are the strongest predictors *and* are unavailable for a brand-new listing. Most projects leak this and never notice. Two model heads = an honest product. |
| 3 | **Leak-aware validation** (`GroupKFold` on `host_id` + spatial block CV) | Multi-listing hosts create near-duplicate rows. Random KFold inflates scores. This is the single biggest silent bug in this dataset. |
| 4 | **NLP that earns its place** (ablation table with CV std bars) | Not "I added BERT". Rather: "text embeddings buy +0.031 R² over TF-IDF, which is 2.4× the fold std → real." Or they don't, and we say so. |
| 5 | **Text price-leakage scrubbing** | Descriptions contain `"$120/night"`, `"€90 per week"`. Un-scrubbed, TF-IDF memorises the target. Free +0.15 R² that is 100% fraudulent. |
| 6 | **Shipped artifact** | FastAPI + Streamlit + Docker + CI. Reproducible with `make all`. Demoable in 60 seconds. |

**Stretch (only if Phase 6 lands early):** train on City A → evaluate zero-shot on City B. Quantifies whether the model learned *pricing* or learned *{CITY}*.

---

## 1. Scope & non-goals

### In scope
- One primary city, one scrape snapshot, `listings.csv.gz` (detailed, ~75 cols) as the spine.
- `reviews.csv.gz` for review-text aggregates (warm model only).
- `neighbourhoods.geojson` for spatial joins.
- OpenStreetMap POIs (transit stops, centre, attractions) for geo features — downloaded once, cached, committed as a small parquet.
- Regression on nightly `price`, plus calibrated 80% intervals.
- Batch training + a small serving API + a demo UI.

### Explicitly out of scope
- `calendar.csv.gz` for modelling (it holds `price` per date → **direct target leak**). Optional Phase 8 use: seasonality *analysis only*, never a feature.
- Occupancy / revenue prediction. Different target, different project.
- Live scraping of Airbnb. Use Inside Airbnb snapshots only.
- Multi-city production service. Cross-city is an *evaluation*, not a deployment.
- Deep learning fine-tuning as the main model. A fine-tuned encoder appears only as a **stacked OOF feature** (Phase 5, optional).

### City selection
| Option | Listings (approx.) | Trade-off |
|---|---|---|
| **London** | ~90k | Best signal-to-noise for embeddings; long runtimes |
| **New York City** | ~40k | English text, strong regulatory outliers (min-nights=30 cliff) — good story |
| **Amsterdam** | ~8–9k | Fast iteration, multilingual text (NL/EN/DE), EUR; small n limits deep features |
| **Barcelona / Paris** | ~18–20k | Good middle ground, multilingual |

**Default: primary = New York City, secondary (cross-city eval only) = London.**
Rationale: large n, English (so a monolingual encoder is a valid ablation arm), and NYC's short-term-rental law creates a genuinely interesting `minimum_nights` structural break to write about. If you want fast iteration on a laptop, swap primary → Amsterdam and secondary → Rotterdam/Brussels; every step below is city-agnostic and driven by `configs/city/*.yaml`.

**Decide now and write it into `configs/city/<name>.yaml`. Do not change it after Phase 3.**

---

## 2. Success criteria (definition of done)

The project is done when **all** of these are true. Not "when the model is good".

### Hard gates
- [ ] `make all` runs end-to-end from a clean clone on a machine with only Docker + `make`, and reproduces the reported metrics to within ±0.002 R².
- [ ] Every number in `README.md` traces to a file in `reports/` produced by a script, not by hand.
- [ ] `pytest` green, `ruff check` clean, CI green on `main`.
- [ ] `git log` contains **zero** references to Claude / AI / co-authors (see §3).
- [ ] Leakage audit script (`make audit`) passes: no feature has |Spearman ρ| with target > 0.95, no `estimated_revenue_*` / `calendar` column present, text scrub verified.

### Model targets (NYC; recalibrate if you switch city)
These are *honest, GroupKFold, leak-free* numbers. They are lower than what you'll see on Kaggle-style notebooks, and that is the point — say so in the README.

| Metric | Naive baseline | Target (cold-start) | Target (warm) |
|---|---|---|---|
| MedAE (median abs error, $) | ~$45 | ≤ $32 | ≤ $28 |
| R² on log1p(price) | ~0.30 | ≥ 0.58 | ≥ 0.64 |
| MedAPE | ~30% | ≤ 22% | ≤ 19% |
| % within ±20% of truth | ~40% | ≥ 55% | ≥ 60% |
| 80% interval coverage | — | 0.78–0.82 | 0.78–0.82 |
| Mean interval width / median price | — | ≤ 0.75 | ≤ 0.65 |

> If your R² comes out ≥ 0.90, **you have a leak.** Stop and run `make audit`. This is a promise, not a warning.

### Deliverables
1. `README.md` — pitch, results table, ablation table, architecture diagram, 60-second demo GIF.
2. `reports/model_card.md` — intended use, limitations, fairness/ethics notes, data provenance.
3. `reports/ablation.md` — the NLP-earns-its-place evidence.
4. `reports/figures/` — SHAP summary, error-vs-price, geo error map, calibration plot.
5. Running demo: `docker compose up` → Streamlit on `:8501`, API on `:8000/docs`.

---

## 3. Repo & attribution policy — **NON-NEGOTIABLE**

**Rule: Abhijit Kumar is the sole author and contributor of every commit, PR, and line of provenance in this repository. No Claude, no AI co-author trailers, no "Generated with" footers, no bot mentions in commit messages, PR bodies, code comments, or docs.**

Claude Code's `includeCoAuthoredBy: false` setting is the documented way to disable the byline, **but it is known to be unreliable** — there are multiple open bug reports of the trailer being appended anyway. So we enforce it in three layers: config, instruction, and a hook that *cannot* be talked around.

### Layer 1 — Claude Code settings (project-scoped, committed)

`.claude/settings.json`:
```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "includeCoAuthoredBy": false
}
```

Also set it globally in `~/.claude/settings.json` so it survives a fresh clone before the project settings load.

### Layer 2 — `CLAUDE.md` at repo root (committed)

Put this as the **first block** of the file, above everything else:

```markdown
# ATTRIBUTION POLICY — HIGHEST PRIORITY, NO EXCEPTIONS

Abhijit Kumar is the ONLY author and contributor of this repository.

NEVER, under any circumstance, in any commit message, PR title, PR body,
issue, code comment, docstring, README, or any other file:
  - add a `Co-Authored-By:` trailer of any kind
  - add "Generated with Claude Code" or any variant
  - add a 🤖 emoji footer or attribution block
  - mention Claude, Anthropic, an AI assistant, or a bot as a contributor

Commit messages are plain Conventional Commits and end at the body.
There is no trailer section. If you are unsure whether to add a line at the
bottom of a commit message: do not add it.

Before running `git commit`, re-read this block.
```

### Layer 3 — `commit-msg` hook (the actual guarantee)

Config and instructions can be forgotten mid-session. A hook cannot. Create `.githooks/commit-msg`:

```bash
#!/usr/bin/env bash
set -euo pipefail
MSG_FILE="$1"

# Case-insensitive scan for any attribution artefact.
if grep -qiE '(co-authored-by|generated with|claude|anthropic|noreply@anthropic|🤖)' "$MSG_FILE"; then
  echo "" >&2
  echo "  COMMIT REJECTED: attribution artefact detected in commit message." >&2
  echo "  This repo has a single author. Strip the offending line and retry." >&2
  echo "" >&2
  grep -inE '(co-authored-by|generated with|claude|anthropic|noreply@anthropic|🤖)' "$MSG_FILE" >&2
  exit 1
fi
```

Wire it up **once, in Phase 0**:
```bash
mkdir -p .githooks
chmod +x .githooks/commit-msg
git config core.hooksPath .githooks     # committed teammates get it via this line in setup
git config user.name  "Abhijit Kumar"
git config user.email "<your-github-noreply-or-real-email>"
```

> `core.hooksPath` is *local config*, not committed. Put the `git config core.hooksPath .githooks` line in `Makefile: setup` and in the README so a fresh clone re-arms it.

### Layer 4 — CI backstop

`.github/workflows/ci.yml` includes a job that scans the **whole history of the PR range**:

```yaml
  attribution-guard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - name: Reject AI attribution in commits
        run: |
          RANGE="${{ github.event.pull_request.base.sha }}..${{ github.sha }}"
          if git log --format='%an%n%ae%n%B' "$RANGE" \
             | grep -iE '(co-authored-by|generated with|claude|anthropic|noreply@anthropic)'; then
            echo "::error::AI attribution found in commit history"; exit 1
          fi
```

### Verification ritual — run before **every** push
```bash
make attribution-check
# = git log --format='%h | %an <%ae> | %s%n%b' -20 | grep -iE 'claude|anthropic|co-authored|generated with' && exit 1 || echo OK
```

### If a bad commit already landed
```bash
# Last commit only:
git commit --amend                       # delete the trailer lines, save
# Deeper (before any push / before anyone pulls):
git rebase -i HEAD~N                     # mark offenders 'reword'
# Nuclear, whole history:
git filter-branch --msg-filter \
  'grep -viE "co-authored-by: claude|generated with \[claude" || true' -- --all
```
Rewriting history is safe **only** while the repo is private and unshared. Do the cleanup before the first public push and this never comes up.

---

## 4. Repository layout

```
pricelens/
├── CLAUDE.md                      # attribution policy + project rules for Claude Code
├── README.md                      # the shop window — written LAST, from reports/
├── Makefile                       # setup | data | features | train | eval | audit | app | all
├── pyproject.toml                 # deps (uv or poetry), ruff + pytest config
├── docker-compose.yml
├── Dockerfile
├── .pre-commit-config.yaml        # ruff, ruff-format, nbstripout, detect-secrets
├── .githooks/commit-msg
├── .github/workflows/ci.yml
│
├── configs/
│   ├── config.yaml                # root config (Hydra-style composition)
│   ├── city/nyc.yaml              # slug, scrape date, urls, currency, centre lat/lon, crs
│   ├── city/london.yaml
│   ├── features/{base,amenities,text_tfidf,text_embed,geo}.yaml
│   └── model/{lgbm,catboost,ridge,quantile}.yaml
│
├── data/                          # .gitignore'd EXCEPT .gitkeep and small cached POIs
│   ├── raw/<city>/<scrape_date>/  # exactly as downloaded, never edited
│   ├── interim/                   # parsed + cleaned parquet
│   ├── processed/                 # feature matrices, fold assignments
│   └── external/pois_<city>.parquet   # committed (small)
│
├── src/pricelens/
│   ├── __init__.py
│   ├── config.py                  # pydantic-settings schema for the YAMLs
│   ├── data/
│   │   ├── download.py            # fetch + checksum + manifest
│   │   ├── load.py                # dtype-safe readers
│   │   └── clean.py               # price parsing, outlier policy, dedup
│   ├── features/
│   │   ├── base.py                # numeric, categorical, date, bathrooms_text
│   │   ├── amenities.py           # normalise → multi-hot → SVD
│   │   ├── text.py                # SCRUB → tfidf/svd, embeddings, handcrafted
│   │   ├── geo.py                 # haversine, POI distances, H3, OOF-KNN target enc
│   │   ├── reviews.py             # warm-only aggregates
│   │   └── pipeline.py            # sklearn ColumnTransformer assembly
│   ├── models/
│   │   ├── baselines.py           # group-median heuristic, ridge
│   │   ├── gbm.py                 # lgbm / catboost wrappers
│   │   ├── quantile.py            # q10/q50/q90 heads
│   │   ├── conformal.py           # split conformal / CQR
│   │   └── stack.py               # OOF blending
│   ├── validation/
│   │   ├── splits.py              # GroupKFold(host_id), spatial block CV
│   │   ├── metrics.py             # log-space + $-space, coverage, width
│   │   └── audit.py               # leakage detector
│   ├── explain/
│   │   ├── shap_utils.py
│   │   └── comps.py               # nearest-neighbour comparable listings
│   └── cli.py                     # typer entrypoints; Makefile just calls these
│
├── apps/
│   ├── api/main.py                # FastAPI: POST /predict, GET /health, /model-info
│   └── ui/streamlit_app.py        # the demo
│
├── notebooks/                     # EDA ONLY. nbstripout enforced. No modelling logic.
│   └── 01_eda.ipynb
├── tests/
│   ├── test_clean.py              # price parser property tests
│   ├── test_text_scrub.py         # THE important one
│   ├── test_splits.py             # no host_id crosses folds
│   ├── test_features.py           # OOF encoders don't see own fold
│   └── test_api.py
├── reports/
│   ├── model_card.md
│   ├── ablation.md
│   ├── metrics/*.json             # machine-written, README reads from these
│   └── figures/*.png
└── models/                        # .gitignore'd; artifacts + MLflow runs
```

**Guardrail:** notebooks import from `src/`, never the reverse. Any logic that appears twice in a notebook gets promoted to `src/` immediately. This is the difference between a portfolio repo and a homework folder.

---

## 5. Phased plan

Each phase has a **gate**. Do not start phase N+1 until phase N's gate is green. Every phase ends with a commit.

### Phase 0 — Scaffold & attribution lock (0.5 day)
**Do**
1. `git init`, set `user.name` / `user.email`, install the `commit-msg` hook, set `core.hooksPath`.
2. Write `CLAUDE.md` (attribution block first, then §7 rules below).
3. `.claude/settings.json` with `includeCoAuthoredBy: false`.
4. `pyproject.toml` (Python 3.11, `uv` recommended), `ruff`, `pytest`, `pre-commit install`.
5. Skeleton `Makefile` with no-op targets, empty `src/pricelens/` package, `tests/test_smoke.py`.
6. CI workflow: lint + test + attribution-guard.

**Gate**
- `make attribution-check` prints OK.
- A deliberate test commit containing `Co-Authored-By: Claude <noreply@anthropic.com>` is **rejected by the hook**. Verify this manually. It is the whole point.
- CI green on the initial commit.

---

### Phase 1 — Data acquisition & contract (0.5 day)
**Do**
1. `src/pricelens/data/download.py`: pull `listings.csv.gz`, `reviews.csv.gz`, `neighbourhoods.geojson` for the chosen city+date from Inside Airbnb into `data/raw/<city>/<date>/`. Write `manifest.json` with URL, SHA256, byte size, row count, download timestamp.
2. **Pin the scrape date.** Inside Airbnb rotates snapshots; an unpinned URL makes your project irreproducible in 3 months. If the pinned quarterly file disappears, keep the SHA in the manifest and document the substitution.
3. Column contract: `configs/city/<city>.yaml` lists expected columns. `load.py` raises on schema drift rather than silently producing NaNs.
4. **Kill-list enforced at load time** — these columns are dropped before anything else touches the frame:
   `calendar_updated`, `estimated_revenue_l365d`, `estimated_occupancy_l365d`, `scrape_id`, `last_scraped`, `source`, `listing_url`, `picture_url`, `host_url`, `host_thumbnail_url`, `host_picture_url`.
   `estimated_revenue_l365d` ≈ price × booked nights. **It is the target multiplied by a constant-ish factor.** Including it gives R² ≈ 0.97 and a worthless model.
5. `notebooks/01_eda.ipynb`: distributions, missingness matrix, price histogram (raw + log), geo scatter, `minimum_nights` cliff, host concentration (`host_id` value counts — motivates GroupKFold).

**Gate**
- `make data` is idempotent; re-running re-verifies checksums, doesn't re-download.
- Row count and column count logged and asserted.
- You can state, out loud, the top 5 data-quality problems in this snapshot. Write them into `reports/model_card.md § Data`.

---

### Phase 2 — Cleaning & target definition (1 day)
This phase decides whether the project is real. See `TECHNICAL_DESIGN.md §3` for exact rules.

**Do**
1. Parse `price`: strip `$`/`,`, cast float. **Note the currency is local** (INR, EUR, ARS…) — record it in the config; never mix currencies across cities without conversion.
2. Target = `log1p(price)`. Train in log space, report in $ space. Back-transform with **Duan's smearing estimator**, not naked `expm1` — see design doc; naked `expm1` biases predictions low.
3. Outlier / filter policy (document each, with the row count dropped):
   - Drop `price <= 0` or null.
   - Drop above the 99.5th percentile *within room_type* (placeholder prices like $9,999 used to hide a listing).
   - Drop `minimum_nights > 30` (these are long-term rentals; different market, and in NYC a regulatory artefact).
   - Drop `accommodates == 0`, `bedrooms > 20`.
   - Flag, don't drop, listings with 0 reviews — that's the cold-start cohort.
4. `bathrooms_text` → `(n_bathrooms: float, is_shared: bool, is_private: bool, is_half_bath: bool)`. Values include `"1 shared bath"`, `"Half-bath"`, `"2.5 baths"`, `""`. Property-test the parser.
5. Percent strings: `host_response_rate`, `host_acceptance_rate` → float.
6. `t/f` → bool: `host_is_superhost`, `instant_bookable`, `has_availability`, `host_identity_verified`.
7. Deduplicate: exact-duplicate rows, and same `host_id` + same lat/lon + same `accommodates` (relist artefacts).
8. Missingness policy: **never impute the mean blindly.** GBMs consume NaN natively — let them. Add `_was_missing` indicators only where missingness is informative (review scores, `host_about`).

**Gate**
- `tests/test_clean.py` passes, including a property test: any string the parser accepts round-trips or is explicitly rejected.
- `reports/metrics/cleaning_log.json` records every filter and its row-loss.
- Total row loss < 12%. If it's more, you're over-filtering; justify it in writing.

---

### Phase 3 — Validation harness **before any model** (0.5 day)
> Building the split *after* the model is how people fool themselves. Build it first.

**Do**
1. `splits.py`:
   - **Primary: `GroupKFold(n_splits=5, groups=host_id)`.** Multi-listing hosts hold near-identical listings at near-identical prices; random KFold puts them on both sides and inflates R² by ~0.05–0.10.
   - **Secondary: spatial block CV.** Assign listings to ~1km H3 cells, group folds by cell. Answers "does it generalise to a neighbourhood it never saw?" Report both.
   - Fixed `seed=42`, folds persisted to `data/processed/folds.parquet`. Every experiment reads the same file. **You have been burned by unseeded runs before — persist the folds to disk, don't regenerate them.**
2. `metrics.py`: RMSE(log), MAE(log), R²(log), and in $-space MedAE, MAE, MedAPE, `pct_within_10`, `pct_within_20`; for intervals: empirical coverage, mean width, normalised width.
3. **Baselines, committed as the floor:**
   - B0: global median.
   - B1: median by `(neighbourhood_cleansed, room_type, accommodates)`, backing off to coarser keys when the cell has < 10 rows. *This is what a smart host does by hand.* If your GBM can't beat it convincingly, the project has no thesis.
   - B2: Ridge on ~15 hand-picked numerics.
4. `audit.py`: the leakage detector (design doc §7).

**Gate**
- `make eval` prints a baseline table.
- `tests/test_splits.py`: assert no `host_id` appears in two folds; assert fold sizes within ±5%.
- Baseline numbers written to `reports/metrics/baselines.json`.

---

### Phase 4 — Feature engineering (2–3 days)
Build in this order, committing each block, measuring CV after each. **Every block must justify itself or get deleted.**

| Block | Files | Adds |
|---|---|---|
| 4a. Base tabular | `features/base.py` | numerics, categoricals, date-derived (`host_tenure_days`, `days_since_last_review`, `listing_age`) |
| 4b. Amenities | `features/amenities.py` | normalise → top-K multi-hot (K≈120) → `amenity_count` → SVD(32) of the full binary matrix |
| 4c. Geo | `features/geo.py` | dist-to-centre, dist-to-nearest-transit, dist-to-top-10-POI, H3 res-8/9 cells, **OOF KNN-mean-log-price (k=10, 25)**, OOF target-encoded neighbourhood |
| 4d. Text — classical | `features/text.py` | **SCRUB first**, then word(1,2) + char(3,5) TF-IDF over `name`+`description`+`neighborhood_overview` → SVD(128); handcrafted: length, caps ratio, `!` count, emoji count, language, keyword flags |
| 4e. Text — neural | `features/text.py` | sentence-transformers embeddings → PCA(64). Monolingual `all-MiniLM-L6-v2` vs multilingual `multilingual-e5-small` as an ablation arm |
| 4f. Reviews (**warm only**) | `features/reviews.py` | last-N review text → mean embedding → PCA(32); review count, recency, velocity, language mix, sentiment |

**Two hard rules for this phase:**
1. **Scrub prices out of text before featurising.** `"Only $95 a night!"` → TF-IDF learns `$95` → the model reads the answer off the page. Regex out currency amounts, standalone numbers ≥ 2 digits adjacent to currency words, and per-night/per-week phrases. `tests/test_text_scrub.py` is mandatory and should be the test you're proudest of.
2. **Every target-derived encoder (KNN-price, neighbourhood target encoding) is computed out-of-fold**, fit on train folds only, with smoothing. In-fold target encoding is the second-most-common silent leak in this dataset after `estimated_revenue`.

**Gate**
- `make features` produces `data/processed/X_{cold,warm}.parquet` deterministically (hash the output; two runs → identical hash).
- `tests/test_features.py` asserts OOF encoders produce NaN/prior for unseen categories and never touch their own fold.
- A CV delta is recorded for each block in `reports/metrics/feature_blocks.json`.

---

### Phase 5 — Modelling (2 days)
**Do**
1. **LightGBM** on log target — the workhorse. Tune with Optuna, ~60 trials, on fold-0-held-out or nested CV. Log everything to MLflow.
2. **CatBoost** with native categorical handling — genuinely different inductive bias, blends well.
3. **Blend**: weights fit on OOF predictions via non-negative least squares. Report blend gain honestly (it's usually +0.005–0.015 — small; say so).
4. **Two heads**:
   - `cold` — trained *without* any review-derived feature. This is the model the product actually serves to a new host.
   - `warm` — everything. This is the model that quantifies "what reviews are worth".
   The gap between them is a **finding**, not a bug. Feature a chart of it.
5. **Quantile heads**: LightGBM `objective="quantile"` at α = 0.1 / 0.5 / 0.9. Guard against quantile crossing (sort the three outputs per row).
6. **Conformal calibration**: split-conformal / CQR on a held-out calibration slice → intervals with a *guaranteed* marginal coverage. Report empirical coverage per fold and per price decile (marginal coverage can hide terrible conditional coverage at the tails — check and disclose).
7. Optional stack: fine-tune a small encoder (`distilbert` / `MiniLM`) with a regression head on text-only → OOF prediction becomes one column in the GBM. Only if Phase 5 is ahead of schedule. Report whether it beat TF-IDF+SVD — often it doesn't on short marketing copy, and that's a *finding worth publishing in the README*.

**Gate**
- Beats B1 by ≥ 25% relative MedAE reduction. If not, stop and debug features — don't tune harder.
- Coverage of the 80% band lands in [0.78, 0.82] on held-out data.
- `make train` reproduces the reported metric within ±0.002 R² on a rerun.
- `make audit` clean.

---

### Phase 6 — Evaluation, ablation, explanation (1.5 days)
This is the phase that turns a model into a project. **Do not skip it to go build the UI.**

**Do**
1. **Ablation table** — the centrepiece of `reports/ablation.md`. Cumulative, each row = mean ± std across the 5 GroupKFolds:

   | Config | R²(log) | MedAE | Δ vs prev | > 1 fold-std? |
   |---|---|---|---|---|
   | B1 group-median | | | — | — |
   | + base tabular | | | | |
   | + amenities | | | | |
   | + geo (incl. OOF-KNN) | | | | |
   | + text handcrafted | | | | |
   | + text TF-IDF/SVD | | | | |
   | + text embeddings | | | | |
   | + review features (warm) | | | | |

   The `> 1 fold-std?` column is what separates you from everyone else. A +0.004 gain with a 0.011 fold-std is **noise**, and saying so out loud is a stronger signal of competence than the gain would have been.
2. **Error analysis** — the part that gets you hired:
   - Error vs. price decile (models under-predict luxury; show it).
   - Error vs. room_type, vs. review_count (cold-start penalty).
   - **Geographic error map** — choropleth of median APE by neighbourhood. Where does it fail? Usually: tourist-core vs. periphery.
   - Worst-20 residuals, read them manually, write a paragraph on what they have in common. This paragraph is worth more than 0.01 R².
3. **SHAP**: global summary (beeswarm) + local waterfall for 3 example listings. `TreeExplainer` on LGBM.
4. **Comparables**: for a query listing, retrieve 5 nearest by a hybrid distance (geo + capacity + text embedding cosine). Grounds the prediction in something a host can verify.
5. **Cross-city zero-shot** (stretch): train NYC → predict London. Expect a large drop. Then: retrain with city-relative features (price percentile within city) and show the drop shrinks. Strong section.

**Gate**
- `reports/ablation.md` complete, with std bars.
- Every figure regenerable via `make eval`.
- You can answer, in one sentence each: *Where does the model fail? Why? What would fix it?*

---

### Phase 7 — Serving & demo (1.5 days)
**Do**
1. **FastAPI** (`apps/api/main.py`):
   - `POST /predict` → pydantic `ListingIn` (validated: lat/lon in city bbox, accommodates 1–16, etc.) → `{point, low, high, currency, model: "cold"|"warm", drivers: [...], comps: [...], model_version, feature_hash}`.
   - `GET /health`, `GET /model-info` (version, train date, city, metrics).
   - Model loaded once at startup, not per request. Pipeline artifact = one joblib containing the full sklearn `Pipeline` so preprocessing can't drift from training.
2. **Streamlit** (`apps/ui/streamlit_app.py`): form → map pin → predicted band as a gauge/range chart, "you are underpriced/overpriced by X%" verdict, SHAP waterfall, comps table with a mini-map.
3. **Docker**: multi-stage build, `docker-compose.yml` running api + ui. Image < 2GB (pin CPU-only torch, or skip torch entirely if embeddings are pre-computed and only the SVD/PCA is needed at inference — prefer this).
4. `tests/test_api.py`: schema validation, boundary cases, a golden-prediction regression test (fixed input → prediction within ±$0.01 of a recorded value — catches silent pipeline drift).

**Gate**
- `docker compose up` from a clean clone → working demo, no manual steps.
- API p95 latency < 300ms on CPU for a single listing.
- Golden test passes.

---

### Phase 8 — Polish (1 day)
- `README.md` last: pitch → demo GIF → results table → ablation → architecture diagram → "how to run" → limitations. Every number pulled from `reports/metrics/*.json`.
- `reports/model_card.md`: intended use, out-of-scope use, training data provenance + scrape date, **limitations** (single snapshot, no seasonality, scraped ≠ transacted price, selection bias toward listed-not-booked), ethics note (Inside Airbnb data is public and collected for housing-policy research; this model must not be used for rent-setting on long-term housing).
- Tag `v1.0.0`. Final `make attribution-check`.

---

## 6. Timeline

| Phase | Days | Cumulative |
|---|---|---|
| 0 Scaffold + attribution lock | 0.5 | 0.5 |
| 1 Data + contract | 0.5 | 1.0 |
| 2 Clean + target | 1.0 | 2.0 |
| 3 Validation harness | 0.5 | 2.5 |
| 4 Features | 3.0 | 5.5 |
| 5 Models | 2.0 | 7.5 |
| 6 Eval + ablation | 1.5 | 9.0 |
| 7 Serving + demo | 1.5 | 10.5 |
| 8 Polish | 1.0 | **11.5** |

~2.5 focused weeks. **If you must cut:** drop 4e (neural embeddings) and the Phase 5 stack before you drop Phase 6. A project with an honest ablation and no BERT is better than one with BERT and no error analysis.

---

## 7. Working rules for Claude Code (put these in `CLAUDE.md` under the attribution block)

```markdown
## Project rules

- Python 3.11. Package manager: uv. Line length 100. Ruff + ruff-format.
- Type hints on every public function. `from __future__ import annotations`.
- No logic in notebooks. Notebooks import from `src/pricelens/`. Never the reverse.
- No hardcoded paths, no hardcoded city names, no magic numbers in `src/`. Everything
  from `configs/`.
- Every stochastic operation takes an explicit `seed` argument. No bare
  `np.random.*`. Folds are read from `data/processed/folds.parquet`, never regenerated.
- Never write a feature that consumes the target without out-of-fold computation.
- Never add a column from the kill-list in configs/city/*.yaml `drop_columns`.
- Write the test before the feature for anything in `features/text.py` and `data/clean.py`.
- One phase = one PR = one focused set of commits. Conventional Commits.
- After changing anything under `features/` or `models/`, run `make audit` and paste
  the result into the PR body.
- If a change makes CV improve by more than 0.05 R² in one step, STOP and assume a
  leak until proven otherwise. Report it, don't celebrate it.
- Do not `pip install` anything not in pyproject.toml. Add it there first.

## Commit message format
    <type>(<scope>): <subject>

    <body — what and why, not how>

  Nothing after the body. No trailers. No footers. See the attribution policy above.
```

---

## 8. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Silent target leak → fake 0.95 R² | **High** | Fatal to credibility | Kill-list at load, `make audit` in CI, "if R² > 0.90 stop" rule |
| Random KFold instead of GroupKFold | High | +0.08 fake R² | `tests/test_splits.py`, folds persisted to disk |
| Price text in descriptions | High | +0.15 fake R² | Scrub + dedicated test file |
| Inside Airbnb rotates/removes snapshot | Medium | Irreproducible | Pin URL + SHA256 in manifest; document substitution if it 404s |
| Embeddings add nothing over TF-IDF | Medium | Deflating | Reframe as a *finding*; the ablation table is the deliverable, not the win |
| Docker image bloat from torch | Medium | Bad DX | Pre-compute embeddings at train time; ship only PCA at inference |
| Scope creep into multi-city serving | Medium | Slips timeline | Cross-city is *eval only*, hard-scoped in §1 |
| Unseeded runs → unreproducible metrics | Medium | Wasted days | Explicit seeds everywhere; `make train` twice → assert same hash |
| Currency confusion across cities | Low | Nonsense numbers | Currency in city config; assert on load |

---

## 9. Kickoff command for Claude Code

Open the repo folder in VS Code, start Claude Code, and give it this:

> Read `PROJECT_PLAN.md` and `TECHNICAL_DESIGN.md` in full before writing any code. We are executing Phase 0 only. Do not start Phase 1.
>
> Phase 0 deliverables: git init with my identity, the `.githooks/commit-msg` hook wired via `core.hooksPath`, `.claude/settings.json`, `CLAUDE.md` (attribution block verbatim from PROJECT_PLAN §3 Layer 2, then the project rules from §7), `pyproject.toml` with the dependency set from TECHNICAL_DESIGN §1, ruff + pytest config, `.pre-commit-config.yaml`, the `src/pricelens/` package skeleton with empty modules matching the §4 tree, a no-op `Makefile` with all targets declared, `tests/test_smoke.py`, and `.github/workflows/ci.yml` including the attribution-guard job.
>
> Then prove the hook works: attempt a commit whose message contains a `Co-Authored-By: Claude` trailer and show me it is rejected.
>
> Stop at the Phase 0 gate and report.
