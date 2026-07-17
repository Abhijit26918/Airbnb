# PriceLens — Technical Design

> Companion to `PROJECT_PLAN.md`. That file is *what and when*; this is *how*.
> Written to be read by Claude Code as implementation context. Code blocks are
> specifications, not copy-paste-ready — they define contracts and gotchas.

---

## 1. Stack & dependencies

| Layer | Choice | Why not the alternative |
|---|---|---|
| Python | 3.11 | 3.12 still has occasional wheel gaps for geo/ML libs |
| Env | `uv` | 10–50× faster than poetry; single lockfile; `uv run` works fine in CI |
| Data | `pandas` 2.x + `pyarrow` | Polars is faster but pandas has the ecosystem for `sklearn`/`shap`; not the bottleneck here |
| Config | `hydra-core` + `pydantic-settings` | Hydra for composition/sweeps, pydantic for validation. Plain YAML+dataclass is acceptable if Hydra feels heavy — but no `argparse` sprawl |
| Models | `lightgbm`, `catboost`, `scikit-learn` | XGBoost adds nothing over LGBM here; skip it |
| Tuning | `optuna` | |
| Tracking | `mlflow` (local file backend) | W&B needs an account; local MLflow keeps the repo self-contained |
| Text | `scikit-learn` TF-IDF, `sentence-transformers`, `langdetect`/`fasttext-langdetect` | |
| Geo | `geopandas`, `shapely`, `h3`, `scikit-learn` BallTree(haversine) | Don't hand-roll haversine loops; BallTree with `metric="haversine"` is the right tool |
| Explain | `shap` | |
| API | `fastapi` + `uvicorn` + `pydantic` v2 | |
| UI | `streamlit` + `pydeck`/`folium` | Gradio is fine; Streamlit has better layout control for a form + map + chart |
| Quality | `ruff`, `pytest`, `pytest-cov`, `hypothesis`, `pre-commit`, `nbstripout` | |
| CLI | `typer` | Makefile targets are thin wrappers over typer commands |

```toml
# pyproject.toml (excerpt)
[project]
name = "pricelens"
requires-python = ">=3.11,<3.12"
dependencies = [
  "pandas>=2.2", "pyarrow>=15", "numpy>=1.26", "scipy>=1.12",
  "scikit-learn>=1.4", "lightgbm>=4.3", "catboost>=1.2", "optuna>=3.6",
  "mlflow>=2.12", "shap>=0.45",
  "sentence-transformers>=2.7", "langdetect>=1.0.9",
  "geopandas>=0.14", "shapely>=2.0", "h3>=3.7,<4",
  "hydra-core>=1.3", "pydantic>=2.7", "pydantic-settings>=2.2", "typer>=0.12",
  "fastapi>=0.111", "uvicorn[standard]>=0.29", "streamlit>=1.35", "pydeck>=0.9",
  "requests>=2.31", "tqdm>=4.66", "joblib>=1.4",
]

[tool.ruff]
line-length = 100
target-version = "py311"
[tool.ruff.lint]
select = ["E","F","I","N","UP","B","SIM","RUF","PD","NPY"]

[tool.pytest.ini_options]
addopts = "-q --strict-markers"
testpaths = ["tests"]
```

**Torch note:** `sentence-transformers` drags in torch (~800MB CPU wheel). Strategy: embeddings are computed **once at feature-build time** and persisted. The serving image needs torch **only** if you accept free text at inference. You do — so either (a) pin `torch` CPU-only via the PyTorch CPU index and accept a ~1.5GB image, or (b) run embedding inference in a separate container and keep the API slim. Choose (a) for simplicity; document the size.

---

## 2. Data contract

### Source
`https://data.insideairbnb.com/<country>/<region>/<city>/<YYYY-MM-DD>/data/listings.csv.gz`
(exact paths are listed on `http://insideairbnb.com/get-the-data` — pin the ones you use)

Files consumed:
| File | Rows | Use |
|---|---|---|
| `listings.csv.gz` (detailed, ~75 cols) | n listings | **The spine.** Everything hangs off this |
| `reviews.csv.gz` (detailed, with `comments`) | ~20–60× n | Warm-model review text only |
| `neighbourhoods.geojson` | ~20–200 polygons | Spatial join, choropleths |
| `calendar.csv.gz` | 365× n | **NOT USED** — contains `price` per date = target leak |

> **Do not use `listings.csv` (the 18-column summary).** It has no text fields. The whole NLP half of the project lives in the 75-column detailed file. Verify on load: `assert df.shape[1] > 60`.

### Column groups (detailed `listings.csv`)

**Target**
- `price` — string, local currency, `"$1,234.00"` format even for EUR/INR cities. The `$` is a formatting artefact; the actual currency is the city's. **Record currency in city config.**

**Identity / grouping**
- `id`, `host_id` ← *`host_id` is the CV grouping key. Non-negotiable.*

**Free text (the NLP surface)**
- `name` — short, marketing-dense, often `"Cozy 1BR · Center · 2 beds"`
- `description` — the main field, 200–1500 chars, HTML entities and `<br />` present
- `neighborhood_overview` — host's pitch for the area; ~40% null
- `host_about` — host self-description; ~50% null
- `amenities` — JSON-ish array-in-a-string: `'["Wifi", "Air conditioning", ...]'`
- `bathrooms_text` — `"1 shared bath"`, `"2.5 baths"`, `"Half-bath"`, `""`
- `property_type` — high cardinality (~80 values), long tail
- `license` — presence/absence is informative (regulatory compliance)

**Numeric / structured**
`accommodates`, `bedrooms`, `beds`, `latitude`, `longitude`, `minimum_nights`, `maximum_nights`, `minimum_minimum_nights`…`maximum_maximum_nights`, `availability_30/60/90/365`, `number_of_reviews`, `number_of_reviews_ltm`, `number_of_reviews_l30d`, `reviews_per_month`, `review_scores_rating` + 6 sub-scores, `host_listings_count`, `host_total_listings_count`, `calculated_host_listings_count*`

**Dates**
`host_since`, `first_review`, `last_review`

**Booleans (`t`/`f` strings)**
`host_is_superhost`, `host_has_profile_pic`, `host_identity_verified`, `instant_bookable`, `has_availability`

**Percent strings**
`host_response_rate`, `host_acceptance_rate` — `"95%"` → 0.95

**Categoricals**
`room_type` (4 values), `neighbourhood_cleansed`, `neighbourhood_group_cleansed`, `host_response_time`, `host_verifications` (list-in-string)

### Kill-list — dropped at load, enforced by config

```yaml
# configs/city/nyc.yaml
drop_columns:
  # LEAKAGE — these encode the target
  - estimated_revenue_l365d      # ≈ price × booked nights. R² 0.97. FATAL.
  - estimated_occupancy_l365d    # denominator of the above; near-leak
  # No signal / metadata
  - scrape_id
  - last_scraped
  - source                       # "neighbourhood search" vs "previous scrape" — scrape artefact
  - calendar_updated             # deprecated, all-null in recent scrapes
  - calendar_last_scraped
  - listing_url
  - picture_url
  - host_url
  - host_thumbnail_url
  - host_picture_url
  - neighbourhood               # free-text, dirty; use neighbourhood_cleansed
  - host_neighbourhood          # ~40% null and inconsistent
```

`estimated_revenue_l365d` is the single most dangerous column in this dataset and it appears in scrapes from ~2024 onward. It is derived from price. If it is in your feature matrix, your model is worthless and every number in your README is a lie. The load-time drop is asserted in `tests/test_clean.py`.

### Schema drift guard

```python
# src/pricelens/data/load.py
REQUIRED = {"id","host_id","price","latitude","longitude","accommodates",
            "room_type","property_type","description","amenities",
            "neighbourhood_cleansed","bathrooms_text","minimum_nights"}

def load_listings(path: Path, cfg: CityConfig) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise SchemaError(f"Snapshot is missing required columns: {sorted(missing)}")
    leaks = set(cfg.drop_columns) & set(df.columns)
    df = df.drop(columns=list(leaks))
    log.info("dropped %d kill-list columns: %s", len(leaks), sorted(leaks))
    assert df.shape[1] > 40, "Looks like the summary listings.csv — need the detailed file"
    return df
```

Fail loud on drift. A silent `NaN` column because Inside Airbnb renamed a field will cost you a day of confused debugging.

---

## 3. Cleaning & target

### 3.1 Price parsing

```python
def parse_price(s: pd.Series) -> pd.Series:
    return (s.astype("string")
             .str.replace(r"[^\d.]", "", regex=True)   # strips $ , and any currency glyph
             .replace("", pd.NA)
             .astype("Float64"))
```
Gotchas: some cities have prices in the millions (IDR, COP) — don't assume a range. Empty strings exist. `"$1,234.00"` → `1234.00`.

### 3.2 Target transform

Price is right-skewed with a long luxury tail. Train on `y = log1p(price)`.

**The bias trap:** `expm1(mean of log predictions) ≠ mean of prices`. Naked back-transform systematically **under-predicts**. Fix with **Duan's smearing estimator**:

```python
# fit on training residuals only
resid = y_train_log - model.predict(X_train)
smearing = np.mean(np.expm1(resid))          # scalar > 0, typically 1.02–1.10

def to_dollars(pred_log, smearing):
    return np.expm1(pred_log) * smearing
```

Report metrics **both** in log space (for model comparison) and dollar space (for the product claim). The dollar-space number is the one a host cares about; the log-space number is the one that's fold-comparable.

> The median-based metrics (MedAE, MedAPE) are the honest headline. Mean-based MAE/RMSE in dollar space are dominated by the handful of $2,000 penthouses and swing wildly across folds.

### 3.3 Filter policy

Every filter is a row in `reports/metrics/cleaning_log.json` with `{rule, rows_before, rows_dropped, pct}`.

```python
FILTERS = [
    ("price_null_or_zero",   lambda d: d.price.notna() & (d.price > 0)),
    ("price_p995_by_room",   lambda d: d.price <= d.groupby("room_type").price.transform(
                                          lambda s: s.quantile(0.995))),
    ("price_floor",          lambda d: d.price >= 10),      # city-dependent; config it
    ("min_nights_le_30",     lambda d: d.minimum_nights <= 30),
    ("accommodates_sane",    lambda d: d.accommodates.between(1, 16)),
    ("bedrooms_sane",        lambda d: d.bedrooms.isna() | (d.bedrooms <= 20)),
]
```

**Rationale, to write in the model card:**
- The p99.5 cap removes *placeholder* prices — hosts who set $9,999 to hide a listing without delisting. These aren't luxury properties; they're not-for-sale. Cap **within room_type**, because $600 is an outlier for a shared room and unremarkable for an entire home.
- `minimum_nights > 30` is a different market (medium-term rental) and, in NYC, a regulatory workaround. Different price dynamics; including them adds a bimodal contaminant.
- **Do not drop zero-review listings.** That's the cold-start cohort — the users of the product.

Total loss should be < 12%. More means you're removing the messiness that makes this project interesting.

### 3.4 `bathrooms_text` parser

```python
BATH_RE = re.compile(r"(?P<num>\d+(?:\.\d+)?)?\s*(?P<kind>half-?bath|shared|private|bath)", re.I)

def parse_bathrooms(s: str | None) -> dict:
    """
    "1 shared bath"   -> {n: 1.0,  shared: True,  private: False, half: False}
    "2.5 baths"       -> {n: 2.5,  shared: False, private: False, half: False}
    "Half-bath"       -> {n: 0.5,  shared: False, private: False, half: True}
    "Shared half-bath"-> {n: 0.5,  shared: True,  private: False, half: True}
    "1 private bath"  -> {n: 1.0,  shared: False, private: True,  half: False}
    None / ""         -> {n: NaN,  shared: False, private: False, half: False}
    """
```
Hypothesis property test: for any input the parser must either return a dict with `n` in `[0, 20] ∪ {NaN}` or raise a typed `ParseError`. Never return a silent wrong number. Known real values also include `"0 baths"` and `"Private half-bath"`.

Fallback: older scrapes have a numeric `bathrooms` column instead. Coalesce.

### 3.5 Missingness

GBMs handle `NaN` natively and use missingness as signal. **Do not impute.** Exceptions:
- Add `review_scores_rating_was_missing` (equivalent to "no reviews" — highly informative).
- Add `host_about_is_empty`, `neighborhood_overview_is_empty`.
- For the Ridge baseline only, median-impute inside the pipeline (fit on train fold).

---

## 4. Feature engineering

### 4.1 Base tabular (`features/base.py`)

Derived:
```
host_tenure_days        = (scrape_date - host_since).days
days_since_first_review = (scrape_date - first_review).days
days_since_last_review  = (scrape_date - last_review).days
listing_activity_span   = (last_review - first_review).days
beds_per_person         = beds / accommodates
bedrooms_per_person     = bedrooms / accommodates
baths_per_person        = n_bathrooms / accommodates
availability_ratio_30   = availability_30 / 30
booking_window          = maximum_nights - minimum_nights   (clip maximum_nights at 1125)
is_multi_listing_host   = calculated_host_listings_count > 1
host_scale_bucket       = pd.cut(calculated_host_listings_count, [0,1,2,5,20,inf])
has_license             = license.notna() & (license != "")
```

`property_type`: ~80 values with a brutal long tail. Keep top 15 by frequency, map the rest to `"Other"`, but **also** keep a boolean `is_entire_unit` derived from the string. CatBoost can take the raw high-cardinality version — that's a legitimate ablation arm.

`maximum_nights` has garbage values (`2147483647`, `99999`). Clip to 1125 (Airbnb's real cap).

### 4.2 Amenities (`features/amenities.py`)

Raw: `'["Wifi", "Air conditioning", "Long term stays allowed", "Shampoo"]'`

```python
def parse_amenities(s: str) -> list[str]:
    try:
        items = json.loads(s)
    except json.JSONDecodeError:
        items = re.findall(r'"([^"]+)"', s or "")
    return [normalise(a) for a in items]

def normalise(a: str) -> str:
    a = unicodedata.normalize("NFKD", a).encode("ascii","ignore").decode()
    a = a.lower().strip()
    a = re.sub(r"\s+", " ", a)
    return AMENITY_ALIASES.get(a, a)
```

**The messiness that matters:** Airbnb amenity strings are not a controlled vocabulary. You'll find `"Wifi"`, `"WiFi"`, `"Wi-Fi"`, `"Fast wifi – 100 Mbps"`, `"Pocket wifi"` as distinct strings — plus brand-name explosions like `"Samsung stainless steel oven"`, `"LG refrigerator"`. Raw one-hot gives you 3,000+ near-duplicate columns of noise.

Two-stage handling:
1. **Alias map** (`configs/features/amenity_aliases.yaml`) — regex → canonical. Cover the top ~200 by frequency; collapse `r".*wifi.*"` → `wifi`, `r".*(oven|stove|refrigerator|fridge).*"` → the appliance class, strip brand tokens.
2. **Then**: top-K=120 multi-hot + `amenity_count` + TruncatedSVD(32) on the *full* (pre-top-K) binary matrix to catch long-tail structure.

Hand-crafted luxury/utility flags worth their own columns: `has_pool`, `has_ac`, `has_free_parking`, `has_paid_parking`, `has_kitchen`, `has_washer`, `has_dryer`, `has_dishwasher`, `has_elevator`, `has_gym`, `has_hot_tub`, `has_workspace`, `is_pets_allowed`, `has_self_checkin`, `has_balcony_or_patio`, `has_waterfront_or_view`.

`amenity_count` alone is a surprisingly strong single feature — it proxies listing effort and property tier. Report it in the SHAP plot.

### 4.3 Geo (`features/geo.py`)

**Distances** (haversine, km):
- `dist_to_centre` — city centre from config.
- `dist_to_nearest_transit` — OSM `public_transport=station` / `railway=subway_entrance`, via BallTree.
- `dist_to_nearest_top_poi` + `n_pois_within_500m` — top attractions from OSM `tourism=attraction`.
- `dist_to_coast_or_river` if the city has one (Amsterdam canals, NYC waterfront — big price effect).

Fetch OSM POIs **once**, cache to `data/external/pois_<city>.parquet`, and **commit it**. It's small (<1MB) and it makes the build reproducible without a live Overpass dependency. Overpass rate-limits and occasionally 504s; do not put it in the critical path of `make features`.

```python
from sklearn.neighbors import BallTree
tree = BallTree(np.radians(poi_coords), metric="haversine")
dist, _ = tree.query(np.radians(listing_coords), k=1)
dist_km = dist[:, 0] * 6371.0
```

**Spatial buckets:** H3 at resolutions 8 (~0.46 km²) and 9 (~0.1 km²). `h3.geo_to_h3(lat, lon, res)`. Res-9 is the useful one for target encoding; res-8 for spatial-block CV.

**OOF KNN price encoding — the single strongest engineered feature, and the easiest to leak:**

```python
def knn_price_features(df, folds, k_list=(5, 10, 25), seed=42):
    """
    For each listing, the mean log-price of its k nearest neighbours,
    computed using ONLY listings in other folds.
    """
    out = {f"knn{k}_logprice": np.full(len(df), np.nan) for k in k_list}
    coords = np.radians(df[["latitude","longitude"]].to_numpy())
    for fold in folds.unique():
        tr = folds != fold
        va = ~tr
        tree = BallTree(coords[tr], metric="haversine")
        for k in k_list:
            d, idx = tree.query(coords[va], k=k)
            out[f"knn{k}_logprice"][va] = df.loc[tr, "y_log"].to_numpy()[idx].mean(axis=1)
            # optionally also: .std(axis=1) -> local price dispersion
    return pd.DataFrame(out, index=df.index)
```

Note what this does **not** do: it never lets a listing see its own fold, and — because folds are grouped by `host_id` — a multi-listing host's other units can't appear as their own neighbours. That combination is what makes the feature legitimate. Get it wrong and you'll see R² jump ~0.15 and feel great about it for two days.

**Also add**: `knn10_logprice_ratio = knn10_logprice - knn10_logprice.mean()` (local price level), and the same KNN restricted to *same room_type* neighbours — that's a much sharper comp.

**Neighbourhood target encoding**: OOF mean of `y_log` per `neighbourhood_cleansed`, with additive smoothing toward the global prior:
```
enc = (sum_y_fold + prior * m) / (count_fold + m),   m ≈ 20
```
Unseen category at inference → the global prior.

**At serving time** the KNN encoder must be fit on the **full training set** (not per-fold) and pickled inside the pipeline. Two different code paths — train-time OOF vs. inference-time full-fit — is the classic place for train/serve skew. Write `tests/test_features.py::test_knn_encoder_train_serve_parity`.

### 4.4 Text — the NLP core (`features/text.py`)

**Fields:** `name`, `description`, `neighborhood_overview`, `host_about`.
Concatenate into `text_all` for the vectorisers; keep per-field handcrafted stats separate.

#### Step 1 — Clean

```python
def clean_text(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)          # stray HTML
    s = re.sub(r"\s+", " ", s).strip()
    return s
```

#### Step 2 — **SCRUB THE PRICE. This is mandatory.**

Descriptions routinely contain the answer:
> `"Great value at $95 per night!"`, `"€1200/month for long stays"`, `"Rate: 150 USD"`, `"weekly discount 20%"`

Un-scrubbed, char n-gram TF-IDF trivially learns `$95` → 95. You get a beautiful R² and a model that fails on any listing whose host didn't paste their price into the description. This is the most instructive bug in the whole project — build the test, then write it up in the README as a section. Reviewers notice.

```python
CURRENCY = r"[$€£₹¥]|\b(usd|eur|gbp|inr|dollars?|euros?|pounds?|rupees?)\b"
NUM      = r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?"

PRICE_PATTERNS = [
    rf"(?:{CURRENCY})\s*{NUM}",                       # $95 / € 1.200
    rf"{NUM}\s*(?:{CURRENCY})",                       # 95 USD
    rf"{NUM}\s*(?:per|/|a)\s*(?:night|nite|day|week|month)",   # 95 per night
    rf"\b(?:price|rate|cost|charge)\w*\b[^.]{{0,20}}?{NUM}",   # "price is 95"
]
_PRICE_RE = re.compile("|".join(PRICE_PATTERNS), re.I)

def scrub_prices(s: str) -> str:
    s = _PRICE_RE.sub(" <PRICE> ", s)
    # residual: any bare 2-4 digit number NOT adjacent to a unit noun
    s = re.sub(rf"(?<!\w){NUM}(?!\s*(?:sqm|sq\.?\s?ft|m2|min|minutes|bed|guest|bath|bedroom))"
               r"(?!\w)", " <NUM> ", s)
    return re.sub(r"\s+", " ", s).strip()
```

Keeping `<PRICE>` as a **token** is deliberate: *whether the host mentions a price at all* is a legitimate, non-leaking feature (it correlates with host sophistication). The magnitude is the leak, not the mention.

Do **not** scrub numbers that carry size/capacity meaning (`"45 sqm"`, `"5 min to metro"`, `"2 bedrooms"`) — hence the negative lookahead. Tune the exclusion list against your city's actual text and record the false-positive rate in a test.

`tests/test_text_scrub.py` — the flagship test file:
```python
@pytest.mark.parametrize("raw,expect_absent", [
    ("Great value at $95 per night!",       ["95"]),
    ("Only 1.200 € a month, long stays",    ["1.200", "1,200"]),
    ("Rate: 150 USD nightly",               ["150"]),
    ("price is 89 for two guests",          ["89"]),
])
def test_removes_prices(raw, expect_absent):
    out = scrub_prices(clean_text(raw))
    for tok in expect_absent:
        assert tok not in out

@pytest.mark.parametrize("raw,expect_present", [
    ("Spacious 45 sqm apartment",  "45"),
    ("5 min walk to the metro",    "5"),
    ("2 bedrooms, 1 bath",         "2"),
])
def test_preserves_semantics(raw, expect_present):
    assert expect_present in scrub_prices(clean_text(raw))
```

**Post-hoc verification, not just unit tests:** after building the TF-IDF matrix, correlate every vocabulary term's presence with `y_log`. Any term with |ρ| > 0.5 gets eyeballed. If a numeric token survives, the scrub has a hole.

#### Step 3 — Handcrafted text features (cheap, interpretable, surprisingly strong)

```
desc_char_len, desc_word_len, name_char_len
overview_is_empty, host_about_is_empty
caps_ratio                # SHOUTY LISTINGS correlate with lower tier
exclamation_count, ellipsis_count
emoji_count               # regex over emoji ranges
n_sentences, avg_word_len
lang                      # langdetect on description; one-hot top 5 + "other"
lang_is_english
n_superlatives            # luxury|luxurious|stunning|spectacular|breathtaking|premium
n_budget_words            # cozy|compact|budget|basic|simple|charming   <- "cozy" means small
n_view_words              # view|skyline|waterfront|canal|park|river|ocean|sea
n_transit_words           # metro|subway|station|tram|bus|central|walk
n_hyperbole               # perfect|amazing|best|incredible|unbeatable
has_price_mention         # did <PRICE> token appear (see above)
```

`"cozy"` is the classic finding: it's a euphemism for small, and it lands as a *negative* SHAP driver. That's a one-line insight that makes the whole NLP section land in a presentation. Look for it.

#### Step 4 — TF-IDF → SVD

```python
word = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_features=50_000,
                       sublinear_tf=True, strip_accents="unicode")
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=5,
                       max_features=50_000, sublinear_tf=True)
# hstack -> TruncatedSVD(n_components=128, random_state=42)
```
Char n-grams matter here: multilingual text, typos, and compound words (Dutch/German) that word tokenisation shreds. Fit inside the CV fold (put it in the sklearn `Pipeline`, not before the split) or you leak vocabulary statistics — a mild leak, but free to avoid.

SVD to 128 because GBMs handle dense low-dim blocks far better than 50k sparse columns.

#### Step 5 — Sentence embeddings

| Model | Dim | Use |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast, English-only. Baseline arm. |
| `intfloat/multilingual-e5-small` | 384 | Multilingual. **Default if your city has non-English listings** (Amsterdam, Barcelona, Paris). Remember E5 wants the `"query: "` / `"passage: "` prefix. |
| `BAAI/bge-small-en-v1.5` | 384 | Stronger English alternative |

→ PCA(64) fitted on train folds. Embed `name + description` and `neighborhood_overview` **separately**; they carry different signal (property vs. location).

Batch on GPU if available; ~40k listings × 384 dims takes ~2 min on a T4, ~15 min on CPU. Cache to parquet keyed by `sha256(scrubbed_text)` so re-runs are free.

**Expected outcome, stated honestly up front:** embeddings often add only +0.005–0.02 R² over TF-IDF+SVD on short marketing copy where the signal is largely keyword-level. If that's your result, **report it as the finding.** "I tested whether semantic embeddings beat lexical features on this text distribution; they didn't, by a margin smaller than the fold std" is a stronger claim than a lucky win, and it's the kind of thing that reads as research maturity rather than tutorial-following.

### 4.5 Reviews — warm model only (`features/reviews.py`)

Join `reviews.csv.gz` on `listing_id`. Take the most recent N=20 comments per listing.

```
n_reviews_total, n_reviews_l12m, n_reviews_l30d
reviews_per_month, days_since_last_review
review_len_mean, review_len_std
review_lang_diversity        # Shannon entropy over detected languages
review_embed_pca_0..31       # mean-pooled embedding of concat(last 20) -> PCA(32)
```

**Why reviews are not leakage but still can't go in the cold model:** a review doesn't contain the price, so including it isn't cheating in the statistical sense. But a brand-new listing has zero reviews. If your only model depends on them, your product cannot serve its actual user. Hence two heads.

The **cold/warm gap is a headline result**, not an inconvenience:
> "Review-derived features add +0.06 R², which is exactly the disadvantage a new host faces. The cold-start model exists so the product degrades gracefully instead of silently."

Serving logic: `n_reviews == 0` → cold model. Else warm. `model_used` is returned in the API response.

---

## 5. Validation

### 5.1 Splits (`validation/splits.py`)

```python
def make_folds(df, n_splits=5, seed=42) -> pd.Series:
    gkf = GroupKFold(n_splits=n_splits)
    folds = pd.Series(-1, index=df.index, dtype=int)
    for i, (_, va) in enumerate(gkf.split(df, groups=df["host_id"])):
        folds.iloc[va] = i
    assert (folds >= 0).all()
    return folds
```
Persist to `data/processed/folds.parquet`. **Every experiment reads this file.** Regenerating folds per run — even with a seed — invites divergence the moment someone changes the row order upstream. (Unseeded run-to-run variance has cost you real debugging time before; this is the structural fix, not a discipline fix.)

`GroupKFold` is deterministic and doesn't shuffle; if you want shuffling use `GroupShuffleSplit` or `StratifiedGroupKFold` (stratify on price decile — better fold balance, recommended).

**Why grouping on `host_id` is non-optional:** in NYC ~30% of listings belong to hosts with 2+ listings; some management companies hold 100+. Their units are near-duplicates at near-identical prices. Random KFold puts twins on both sides of the split and the model "predicts" by recall. Measured inflation on this dataset: **+0.05 to +0.10 R²**. Every notebook that reports 0.72 without grouping is reporting ~0.64.

**Secondary — spatial block CV**: group by H3 res-8 cell. Answers the harder generalisation question. Expect scores to *drop* — geo features are doing heavy lifting and this split takes their crutch away. Report both; the delta between them is itself a result.

Write `reports/metrics/split_comparison.json` with `{random_kfold, group_kfold, spatial_block}` R². Putting the inflated random-KFold number in your README **next to** the honest one, labelled, is a flex.

### 5.2 Metrics (`validation/metrics.py`)

```python
def regression_metrics(y_log, pred_log, smearing=1.0) -> dict:
    y, p = np.expm1(y_log), np.expm1(pred_log) * smearing
    ape = np.abs(p - y) / np.clip(y, 1, None)
    return {
        "rmse_log": rmse(y_log, pred_log),
        "mae_log": mae(y_log, pred_log),
        "r2_log": r2_score(y_log, pred_log),
        "medae": float(np.median(np.abs(p - y))),
        "mae": float(np.mean(np.abs(p - y))),
        "medape": float(np.median(ape)),
        "pct_within_10": float((ape <= 0.10).mean()),
        "pct_within_20": float((ape <= 0.20).mean()),
    }

def interval_metrics(y, lo, hi) -> dict:
    return {
        "coverage": float(((y >= lo) & (y <= hi)).mean()),
        "mean_width": float(np.mean(hi - lo)),
        "norm_width": float(np.mean(hi - lo) / np.median(y)),
    }
```

Report **mean ± std across folds**, always. A single number with no dispersion is uninterpretable and every ablation claim depends on the std.

### 5.3 Leakage audit (`validation/audit.py`) — `make audit`, runs in CI

```python
CHECKS = [
  "no kill-list column present in X",
  "no feature has |spearman| with y > 0.95",
  "no feature name matches (revenue|occupancy|calendar|adjusted_price)",
  "scrubbed text contains no token with |corr to y| > 0.5",
  "folds: zero host_id overlap between train/val",
  "OOF encoders: refit on a shuffled y -> encoder-derived feature importance collapses",
  "duplicate rows: 0",
]
```

The shuffled-`y` check is the strong one and worth the effort: retrain the full pipeline on a **randomly permuted target**. If R² isn't ≈ 0, something in your pipeline is reading the target outside the fold. It's a 10-minute run that catches leaks no eyeball review will.

---

## 6. Models

### 6.1 Baselines (`models/baselines.py`)

```python
class GroupMedianBaseline:
    """B1 — what a competent host does by hand.
    Backs off: (nbhd, room_type, accommodates) -> (nbhd, room_type) -> (room_type) -> global
    Any cell with n < 10 backs off one level."""
```
This is the benchmark that matters. Beating a global median is meaningless; beating a well-constructed group median by 25%+ MedAE is a real claim.

### 6.2 LightGBM (primary)

```yaml
# configs/model/lgbm.yaml
objective: regression          # L2 on log target
metric: rmse
n_estimators: 3000
learning_rate: 0.03
num_leaves: 64
max_depth: -1
min_child_samples: 30
feature_fraction: 0.7
bagging_fraction: 0.8
bagging_freq: 1
lambda_l1: 0.1
lambda_l2: 1.0
early_stopping_rounds: 200
seed: 42
n_jobs: -1
verbosity: -1
```

Categorical handling: pass `categorical_feature=[...]` with pandas `category` dtype. **Warning:** LightGBM's categorical splits overfit high-cardinality columns badly. For `neighbourhood_cleansed` prefer the OOF target encoding; reserve native categorical for `room_type`, `host_response_time`, `property_type_top15`.

`objective="regression"` (L2) vs `"huber"`/`"regression_l1"`: L2 on the log target is already robust-ish because log compresses the tail. Try `huber` as an ablation; it usually helps MedAE slightly and hurts R². Pick by the metric you're claiming.

Optuna: 60 trials over `num_leaves`, `min_child_samples`, `feature_fraction`, `lambda_l2`, `learning_rate`. Tune on fold 0's inner split, evaluate on all 5 — **do not** tune on the same folds you report, or your reported number is optimistic by ~0.01.

### 6.3 CatBoost (secondary)

Native categorical handling with ordered target statistics — genuinely different bias, blends well with LGBM (OOF correlation ~0.97, which still leaves useful diversity).

```yaml
iterations: 3000
learning_rate: 0.03
depth: 8
l2_leaf_reg: 3.0
loss_function: RMSE
random_seed: 42
od_type: Iter
od_wait: 200
```

### 6.4 Blending (`models/stack.py`)

```python
from scipy.optimize import nnls
W, _ = nnls(oof_matrix, y_log)     # oof_matrix: (n, n_models)
W = W / W.sum()
```
Fit weights on OOF preds only. Expect +0.005–0.015 R². **Report the gain honestly** — if it's inside the fold std, say the blend isn't justified and ship the single model. Shipping the simpler model with a written reason is a better look than a 0.4% ensemble gain.

### 6.5 Quantile heads (`models/quantile.py`)

```python
for alpha in (0.1, 0.5, 0.9):
    LGBMRegressor(objective="quantile", alpha=alpha, **params)
```
Three separate models. **Quantile crossing** (q10 > q90 on some rows) happens — fix by row-wise sorting the three outputs. Report the crossing rate before the fix; it's typically 0.5–2% and it's an honest detail.

### 6.6 Conformal calibration (`models/conformal.py`)

Quantile regression gives *approximate* coverage. Conformal gives a **finite-sample marginal guarantee** under exchangeability. This is the section that makes the project feel like it was built by someone who reads papers.

**Conformalized Quantile Regression (CQR)** — Romano, Patterson & Candès (2019):

1. Split: proper-train / calibration (e.g. 80/20 of the training folds, **grouped by host_id** so a host doesn't straddle the split).
2. Fit q̂_lo (α/2) and q̂_hi (1−α/2) on proper-train.
3. On calibration set compute conformity scores:
   `E_i = max(q̂_lo(x_i) − y_i, y_i − q̂_hi(x_i))`
4. `Q = ceil((n_cal+1)(1−α)) / n_cal` empirical quantile of `E`.
5. Interval: `[q̂_lo(x) − Q, q̂_hi(x) + Q]`.

Guarantees `P(y ∈ C(x)) ≥ 1−α` **marginally**.

**The caveat you must disclose:** marginal coverage says nothing about conditional coverage. Your 80% band can be 95% covering for $80 studios and 40% covering for $600 penthouses while averaging 80%. **Report coverage per price decile and per room_type.** If it's badly non-uniform, mention Mondrian/group-conditional conformal (calibrate `Q` separately within each room_type) as the fix — and implement it if time allows. That paragraph alone is a differentiator.

Do the conformal work in **log space**, then back-transform the interval endpoints (monotone transform → interval is preserved; smearing does **not** apply to quantiles, only to the conditional mean — getting this right is a subtle point worth a footnote).

### 6.7 Optional: text-stack

Fine-tune `distilbert-base-uncased` (or `MiniLM`) + regression head on scrubbed `name+description` → `y_log`. Generate 5-fold OOF predictions → add as a single column `text_nn_pred` to the GBM feature matrix.

This is the "proper NLP" arm. Budget: ~30 min/fold on a T4. Only after Phase 6 is done. If it doesn't beat TF-IDF+SVD as a feature, that's a publishable-in-README negative result — short marketing copy is lexically saturated and there isn't much compositional semantics left for a transformer to extract.

---

## 7. Explainability

### SHAP
`shap.TreeExplainer(lgbm)` on a 2000-row sample for the global beeswarm. Group the 128 SVD text dims into one aggregate bar (`sum(|shap|)` over the block) — 128 individually meaningless columns in a beeswarm is noise, and the aggregated block bar is what tells the "text is worth X% of the decision" story.

For the API: `shap_values` for the single row → top 5 by |value| → human-readable strings:
```
"Entire home (vs. private room)      +$34"
"3 blocks from a subway stop         +$12"
"'cozy' in description               −$8"
"Amenity count: 41 (top quartile)    +$9"
```
Map raw feature names → human labels via `configs/features/labels.yaml`. Never surface `knn10_logprice` to a host.

### Comparables (`explain/comps.py`)
Hybrid distance over the training set:
```
d = w_geo * haversine_km_normalised
  + w_cap * |accommodates_i - accommodates_q| / 4
  + w_txt * (1 - cosine(embed_i, embed_q))
  + w_type * (room_type_i != room_type_q)
```
Weights in config (start `0.4 / 0.2 / 0.3 / 0.1`). Return top 5 with their actual prices. Comps are the trust mechanism — a host who disbelieves the model can check the comps and either update or find the flaw. Both outcomes are good.

---

## 8. Serving

### API contract

```python
class ListingIn(BaseModel):
    latitude: float = Field(..., ge=CITY_BBOX.lat_min, le=CITY_BBOX.lat_max)
    longitude: float = Field(..., ge=CITY_BBOX.lon_min, le=CITY_BBOX.lon_max)
    room_type: Literal["Entire home/apt","Private room","Shared room","Hotel room"]
    property_type: str
    accommodates: int = Field(..., ge=1, le=16)
    bedrooms: float | None = Field(None, ge=0, le=20)
    beds: float | None = Field(None, ge=0, le=40)
    bathrooms_text: str | None = None
    amenities: list[str] = []
    name: str = ""
    description: str = ""
    neighborhood_overview: str | None = None
    minimum_nights: int = Field(1, ge=1, le=30)
    instant_bookable: bool = False
    host_is_superhost: bool = False
    number_of_reviews: int = Field(0, ge=0)     # 0 -> cold model
    review_scores_rating: float | None = Field(None, ge=0, le=5)

class Driver(BaseModel):
    label: str
    impact_currency: float
    direction: Literal["up","down"]

class Comp(BaseModel):
    id: int; price: float; distance_km: float
    room_type: str; accommodates: int; url: str

class PredictionOut(BaseModel):
    point: float
    low: float                 # conformalised 10th
    high: float                # conformalised 90th
    currency: str
    model_used: Literal["cold","warm"]
    confidence: Literal["high","medium","low"]   # from interval width vs. point
    drivers: list[Driver]
    comps: list[Comp]
    model_version: str
    trained_on: str            # scrape date
```

`GET /health` → `{status, model_version, uptime_s}`
`GET /model-info` → training city, scrape date, row count, headline metrics, feature count.

**Load the model once at startup** via a lifespan context, not per request. **Ship the entire sklearn `Pipeline`** (preprocessing + model) in one joblib. If preprocessing lives in the API code instead of the pipeline, it *will* drift from training and you will ship a silently-wrong model. This is the most common failure mode of ML services.

`feature_hash` in the response = hash of the ordered feature-name list. If it doesn't match what the model was trained with, refuse to serve. Cheap insurance.

### Streamlit UI
Layout: left column = form + map pin (`st.map` / `pydeck` click); right column = the band chart (point marker on a horizontal range), the verdict banner (`"€99 is 24% below the predicted band → you're likely leaving money on the table"`), SHAP waterfall, comps table + mini-map.

The verdict banner is the demo. It's the thing that makes a non-technical viewer immediately understand why the project exists. Make it big.

### Docker
Multi-stage: builder (uv sync) → runtime (slim, non-root, no build deps). Pin CPU torch:
```dockerfile
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```
`docker-compose.yml` with `api` (8000) and `ui` (8501), `ui` depends_on `api`, model artifacts mounted or baked in.

---

## 9. Testing strategy

| File | Covers | Notes |
|---|---|---|
| `test_clean.py` | price parser, bathrooms parser, filters, kill-list drop | Hypothesis property tests on both parsers |
| `test_text_scrub.py` | **price scrubbing** | The flagship. Positive + negative cases. Add every real-world string you find in EDA |
| `test_splits.py` | no host_id crosses folds; fold sizes ±5%; determinism | |
| `test_features.py` | OOF encoders don't see own fold; unseen category → prior; train/serve KNN parity; feature-matrix hash stable across runs | |
| `test_metrics.py` | metric math on synthetic data with known answers | |
| `test_api.py` | schema validation, bbox rejection, cold/warm routing, **golden prediction** | Golden test = fixed JSON in → prediction within ±$0.01 of a recorded value. Catches pipeline drift |
| `test_audit.py` | shuffled-y → R² ≈ 0 | Marked `@pytest.mark.slow`, runs nightly not per-commit |

Coverage target: ≥80% on `src/pricelens/data` and `src/pricelens/features`. Don't chase coverage on `apps/`.

---

## 10. Reproducibility

- **Seeds:** one `seed` in root config → threaded into every model, every SVD/PCA, every sampler. No bare `np.random.*`, no unsedeed `train_test_split`. Add `PYTHONHASHSEED=0` in the Makefile.
- **Folds on disk**, never regenerated (§5.1).
- **Data manifest** with SHA256 per raw file; `make data` verifies and refuses to proceed on mismatch.
- **Feature-matrix hash** logged after `make features`. Two runs → identical hash, asserted in a test.
- **MLflow** local file store: params, metrics, feature list, config snapshot, git SHA per run.
- **Model artifacts** are `{pipeline, smearing, conformal_Q, feature_names, feature_hash, model_version, train_meta}` in a single joblib.
- `make all` = `setup data clean features train eval audit` — clean clone → reported numbers.

---

## 11. Known traps — the checklist

Read this before every phase gate.

1. **`estimated_revenue_l365d`** → R² 0.97. The dataset ships you the answer. Kill-list it. *(Present in scrapes from ~2024 on.)*
2. **`calendar.csv` price** → same, more obviously. Not used.
3. **Random KFold with multi-listing hosts** → +0.05–0.10 fake R².
4. **Price strings inside `description`** → +0.15 fake R² via char n-grams.
5. **In-fold target encoding** (neighbourhood, KNN price) → silent, large, feels like a win.
6. **Naked `expm1`** → systematic under-prediction. Use smearing.
7. **Summary `listings.csv` (18 cols)** instead of detailed (75 cols) → no text, no project.
8. **`maximum_nights` = 2147483647** → int overflow garbage. Clip at 1125.
9. **`price` is local currency** despite the `$` glyph. Never mix cities without conversion.
10. **Scraped price ≠ transacted price.** It's an *asking* price, possibly for an empty listing nobody books. Your model predicts what hosts *ask*, not what the market *clears*. **State this in the model card.** It's the single biggest honest limitation of the project and volunteering it is worth more than hiding it.
11. **Single snapshot = no seasonality.** A July NYC scrape prices differently than a January one. Scope note in the model card.
12. **Selection bias:** listings that exist on Airbnb at scrape time. Delisted/failed listings are invisible. Survivorship.
13. **Tuning on the reported folds** → ~+0.01 optimism. Use nested or a held-out tuning fold.
14. **Marginal conformal coverage hides conditional failure.** Report per-decile.
15. **Preprocessing outside the pipeline** → train/serve skew. Ship one joblib.
16. **`bathrooms` vs `bathrooms_text`** — older scrapes have the numeric column, newer the text one. Coalesce, don't assume.

---

## 12. README skeleton (write this last, from `reports/`)

```markdown
# PriceLens — what should this Airbnb listing cost?
> [60-second demo GIF]

Predicts a defensible nightly price *band* for {CITY} listings from structure,
geography, amenities, and free text — with the drivers and the comps that justify it.

## Results
[table from reports/metrics/final.json — cold + warm, vs. group-median baseline]

## Does the NLP actually help?
[ablation table with fold stds; the honest answer, whatever it is]

## What I found
- "cozy" costs you $X.
- Review features are worth +0.06 R² — which is the exact penalty a new listing pays.
- Random KFold reports 0.71 on this data. Grouped by host, it's 0.64. The gap is
  30% of hosts owning multiple near-identical units.
- [the paragraph from the worst-20 residual read-through]

## Architecture
[diagram]

## Run it
    make all          # clean clone -> reported numbers
    docker compose up # demo on :8501

## Limitations
[the honest list — scraped ≠ transacted, single snapshot, selection bias, coverage caveats]
```

The "What I found" section is what people actually read. Write it like you're telling someone something they didn't know, not like you're listing what you did.
