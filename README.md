# Tanzania Tourism Expenditure Classifier

## Project overview

This project builds a machine learning model that classifies the range of
expenditure a tourist spends while visiting Tanzania into one of **six
expenditure bands**:

- Low Cost
- Lower Cost
- Normal Cost
- High Cost
- Higher Cost
- Highest Cost

The dataset contains 24,675 survey responses collected by Tanzania's National
Bureau of Statistics, covering visitor demographics (age group, country of
origin), trip characteristics (purpose of visit, main activity, tour
arrangement, travel companions), and package inclusions (transport,
accommodation, food, sightseeing, guided tours, insurance).

Predictions are submitted as **class probabilities across all six bands**
and evaluated with **multi-class log loss**, not accuracy — so producing
well-calibrated probabilities matters as much as picking the most likely
class. A model that is confidently wrong is penalized far more heavily than
one that hedges appropriately across plausible classes.

Tourism contributes approximately 17% of Tanzania's GDP. A useful model
could help tour operators and the Tanzania Tourism Board give visitors
up-front expenditure estimates before they travel.

This challenge is hosted by **Briisp Academy**, part of the AI4D Africa
Anglophone Multidisciplinary Research Lab. It is a university hackathon open
to English-speaking African countries and ran from **September 9 to 12,
2026**.

## Project structure

```
.
├── data/
│   └── raw/                       # Train.csv, Test.csv, SampleSubmission.csv
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   ├── 02_preprocessing.ipynb          # Data cleaning and feature preparation
│   ├── 03_baseline_model.ipynb         # Initial benchmark model
│   ├── 04_modelling.ipynb              # Model training and improvement
│   ├── 05_submission.ipynb             # Prediction and submission preparation
│   ├── ...                             # (fill in any intermediate notebooks here)
│   ├── 09_refined_model.ipynb          # Refined CatBoost pipeline (see below)
│   └── 10_optimized_model.ipynb        # CatBoost + LightGBM blend, smoothed encoding (see below)
├── .gitignore                     # Files excluded from version control
├── .python-version                # Project Python version
├── Makefile                       # Common project commands
├── main.py                        # Project entry point
├── pyproject.toml                 # Project metadata and direct dependencies
├── requirements.txt               # Reproducible pinned environment
├── README.md                      # Project documentation
└── uv.lock                        # Locked dependency resolution
```

> **Note:** this table currently jumps from `05_submission.ipynb` to
> `09_refined_model.ipynb`. If notebooks 06–08 exist in your working
> directory, add a one-line description of each here so the structure stays
> an accurate map of the repo.

The notebooks are intended to be run in order: understand the data, prepare
features, establish a baseline, train candidate models, and generate the
final submission. `10_optimized_model.ipynb` is the current best-performing
pipeline; `09_refined_model.ipynb` is the CatBoost-only pipeline it builds
on. Both are described in detail below.

## Modeling approach (`09_refined_model.ipynb`)

This section explains what the refined pipeline actually does, end to end,
so the modeling choices are traceable without having to read the notebook
cell by cell.

### 1. Data loading

Train, test, and sample-submission CSVs are loaded from `data/raw/` if
present locally, falling back to the versioned copies on GitHub. This keeps
the notebook runnable both locally and in a fresh Colab/CI environment
without manual file uploads.

### 2. Target setup

The six expenditure bands are mapped to integer indices 0–5
(`class_to_idx` / `idx_to_class`), since CatBoost's `MultiClass` objective
trains on integer-encoded labels internally. The original string labels are
recovered at submission time.

### 3. Feature engineering

Raw survey fields are cleaned and expanded into a richer feature set:

- **Typo/missing-value fixes** — e.g. `"Widlife Tourism"` → `"Wildlife
Tourism"`; `travel_with` missing values are treated as `"Alone"` rather
  than dropped, since "traveling alone" is itself informative for cost.
- **Group-size aggregates** — `total_people` (female + male count) and
  `total_nights` (mainland + Zanzibar nights), each with a "safe" version
  that floors zero at one to avoid divide-by-zero in the ratio features
  below.
- **Ratio features** — `mainland_ratio`, `zanzibar_ratio`, `female_ratio`,
  and `nights_per_person`. These capture _how_ a trip is structured (e.g. a
  short mainland-only solo trip vs. a long family trip split across both
  regions), which is more predictive of spend than the raw counts alone.
- **Package depth metrics** — `package_count` (how many of the seven
  package inclusions — transport, accommodation, food, sightseeing, guided
  tour, insurance — were marked "Yes"), plus `package_depth` (that count
  normalized to a 0–1 scale), `is_full_package`, and `has_no_package` flags.
  A tourist who pre-paid for everything has a fundamentally different cost
  profile than one paying à la carte.

### 4. Categorical handling

Fifteen categorical fields (`country`, `age_group`, `travel_with`,
`purpose`, `main_activity`, `info_source`, `tour_arrangement`, the seven
package Yes/No columns, and `first_trip_tz`) are cast to string and passed
natively to CatBoost via its `cat_features` argument. CatBoost handles
categorical splits internally using ordered target statistics, so these
columns don't need one-hot or label encoding.

On top of that, three high-cardinality columns get **frequency encoding**
(`country_freq`, `purpose_freq`, `main_activity_freq`) — a numeric feature
equal to how often each category value appears in the data. This gives the
model a cheap, monotonic signal for "how common is this value," which
complements CatBoost's own categorical handling.

### 5. Out-of-fold target encoding

`country` and `purpose` additionally get **per-class likelihood encoding**:
for each of the six target classes, what fraction of rows with a given
`country` (or `purpose`) value belong to that class. This is computed
**inside each cross-validation fold, using only that fold's training
split** — never the validation or test rows — which is what makes it
leakage-free. Encoding maps are fit fold-by-fold, then applied to that
fold's train, validation, and test data.

### 6. Cross-validation and model training

Training uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
so each fold preserves the overall class distribution. For each fold, a
`CatBoostClassifier` is trained with:

| Parameter                       | Value        | Why                                                                                                                                     |
| ------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `loss_function` / `eval_metric` | `MultiClass` | Directly optimizes and monitors the competition metric family (multi-class log loss).                                                   |
| `iterations`                    | 1800         | Generous ceiling; early stopping decides the actual number used.                                                                        |
| `learning_rate`                 | 0.03         | Slow enough for stable convergence over 1800 rounds.                                                                                    |
| `depth`                         | 6            | Moderate tree depth — enough to capture interactions without overfitting a ~25k-row dataset.                                            |
| `l2_leaf_reg`                   | 5            | L2 regularization on leaf values, guards against overfitting on high-cardinality categoricals.                                          |
| `random_strength`               | 0.8          | Adds randomness to split scoring, another overfitting guard.                                                                            |
| `early_stopping_rounds`         | 120          | Training stops if validation `MultiClass` loss hasn't improved in 120 rounds; the best iteration is restored via `use_best_model=True`. |

Out-of-fold (OOF) predictions accumulate one prediction per training row
(from the fold where it was held out), and test predictions are averaged
across all five folds' models.

### 7. Probability calibration

Raw predicted probabilities are clipped to `[1e-15, 1 - 1e-15]` and
renormalized to sum to 1 per row. This exists purely as a numerical safety
net: log loss penalizes a confident wrong prediction (probability near 0
for the true class) extremely harshly, and clipping caps how much damage
any single misprediction can do without changing well-calibrated
predictions in any meaningful way.

### 8. Evaluation

The final OOF log loss is computed with `sklearn.metrics.log_loss` across
all five folds' held-out predictions — a robust estimate of how the model
would score on unseen data, since no row is ever scored by a model that
trained on it.

### 9. Submission

Test predictions are assembled into a DataFrame with the six class
probability columns, then **reindexed to `SampleSubmission.csv`'s exact
column order** and **merged on `Tour_ID` against the sample submission's ID
column** — not just concatenated — so the row order and coverage are
guaranteed to match what the competition platform expects, even if the
model's internal row order differs.

## Modeling approach (`10_optimized_model.ipynb`)

This notebook takes `09_refined_model.ipynb` as its starting point and
targets the specific weaknesses that hurt log loss most: overconfident
probabilities from noisy encodings, a subtle test-set leak, and relying on
a single model family. The data loading, target setup, base feature
engineering, and categorical feature list are unchanged from `09` (see
above) — everything below is what's different.

### 1. Additional feature engineering

On top of the ratio and package-depth features from `09`, three more
features are added:

- **Log-transforms** — `log_total_nights` and `log_total_people`
  compress the long right tail of group size and trip length, which helps
  tree splits find thresholds in the densely populated low end of the
  range rather than wasting splits on rare large-group outliers.
- **`nights_x_people`** — an explicit interaction term, since a long trip
  with many people scales cost differently than either factor alone.
- **`age_group_ordinal`** — the leading number is parsed out of each
  `age_group` bucket (e.g. `"25-44"` → `25`) and kept alongside the raw
  categorical. This gives the model an ordinal signal ("older/younger
  than") in addition to CatBoost's categorical split logic, which treats
  age brackets as unordered by default.
- **`purpose_x_arrangement`** — a string concatenation of `purpose` and
  `tour_arrangement` (e.g. `"Business_Independent"` vs.
  `"Business_Package"`), added as its own native categorical feature. The
  same purpose can have a very different cost profile depending on whether
  it was independently arranged or packaged.

### 2. Leak-free frequency encoding

In `09`, `country_freq` / `purpose_freq` / `main_activity_freq` were built
from `pd.concat([train, test])` value counts — which means the encoding
for every row, including training rows, is influenced by how often that
category appears in the _test_ set. `10` fixes this: frequency maps are
fit on the training data only, then applied to test (with `.fillna(0)` for
any category that appears in test but never in train). This removes a
subtle information leak, at the cost of frequency values for train-unseen
test categories now being present but zero.

### 3. Smoothed (Bayesian) target encoding

`09`'s per-class likelihood encoding is exact but unstable for rare
categories — a country with two rows, one of which is "Highest Cost,"
encodes as a 50% likelihood for that class, which is a wildly
overconfident estimate that log loss will punish hard if it's wrong on new
data. `10` replaces this with **shrinkage toward the global class prior**:

```
shrink   = count / (count + SMOOTHING_K)
smoothed = raw_likelihood * shrink + global_class_prior * (1 - shrink)
```

A category seen thousands of times keeps close to its raw likelihood
(`shrink → 1`); a category seen only once or twice gets pulled most of the
way back toward the overall class distribution (`shrink → 0`).
`SMOOTHING_K = 15` sets how many observations it takes before a category's
own signal starts to dominate the prior — worth tuning down if your
high-cardinality columns have thin per-category volume, or up if you still
see the encoding overfitting rare values. This still runs **inside each
CV fold on the training split only**, so it stays leakage-free.

### 4. Multi-seed CatBoost bagging

Each fold now trains **three CatBoost models** (`random_seed` values 42,
202, 777) instead of one, and averages their predicted probabilities
before recording the fold's OOF and test predictions. CatBoost's own
training stochasticity (random feature/split sampling) means a single
seed's model is one noisy sample of what the model could learn; averaging
three seeds reduces that variance without changing the hyperparameters at
all. A `bagging_temperature=0.3` parameter is also added to control the
strength of CatBoost's Bayesian bootstrap sampling.

### 5. LightGBM as a second model, blended by OOF search

A `LGBMClassifier` is trained on the same folds, same features, and same
in-fold target encoding as CatBoost (categorical columns are cast to
pandas `category` dtype with a shared category set across train/val/test
so LightGBM's native categorical handling sees consistent codes). CatBoost
and LightGBM build trees differently — CatBoost grows balanced/oblivious
trees with ordered boosting, LightGBM grows leaf-wise trees — so their
errors tend to be only partially correlated. Blending two models that make
different mistakes generally beats either model alone on log loss.

The blend weight isn't fixed: after both models finish CV, the notebook
grid-searches a CatBoost weight `w` from 0.0 to 1.0 in steps of 0.05,
computing OOF log loss for `w * catboost + (1 - w) * lightgbm` at each
step, and keeps whichever `w` scores lowest. This weight is then applied
identically to the test-set blend, so the reported OOF score and the
submitted predictions use the same mixing ratio.

### 6. Calibration and evaluation

The final blended probabilities go through the same clip-and-renormalize
step as `09` (`[1e-15, 1 - 1e-15]`, rows resummed to 1), and the final OOF
log loss is reported on the calibrated blend — not on either model
individually — since that blended, calibrated score is what the submission
file actually reflects.

### 7. Submission

Identical to `09`: predictions are reindexed to `SampleSubmission.csv`'s
column order and merged on `Tour_ID` against the sample submission's ID
column, guaranteeing row order and coverage regardless of internal
processing order. Output file: `optimized_submission.csv`.

### Runtime note

`10` trains 3 CatBoost seeds × 5 folds plus 5 LightGBM folds — roughly 3×
the training cost of `09`. It also requires `lightgbm` in addition to
`catboost` (already listed in `requirements.txt`).

## Libraries

`requirements.txt` contains the pinned runtime environment. The packages
listed below include the project's main libraries and their supporting
dependencies.

### Main machine learning and data libraries

| Library        | Version | Purpose                                                                                    |
| -------------- | ------- | ------------------------------------------------------------------------------------------ |
| `catboost`     | 1.2.10  | Gradient boosting for classification, with strong native support for categorical features. |
| `lightgbm`     | 4.7.0   | Fast, memory-efficient gradient boosting for tabular data.                                 |
| `numpy`        | 2.4.6   | Numerical arrays and vectorized mathematical operations.                                   |
| `pandas`       | 3.0.5   | Loading, cleaning, transforming, and analyzing tabular survey data.                        |
| `scikit-learn` | 1.9.0   | Preprocessing, train/validation utilities, metrics, and machine learning workflows.        |

### Visualization and notebook support

| Library      | Version | Purpose                                                                     |
| ------------ | ------- | --------------------------------------------------------------------------- |
| `matplotlib` | 3.11.1  | General-purpose plotting for exploratory analysis.                          |
| `plotly`     | 7.0.0   | Interactive charts and visual data exploration.                             |
| `graphviz`   | 0.21    | Python interface for rendering graph and tree diagrams.                     |
| `pillow`     | 12.3.0  | Image loading and processing used by visualizations and notebooks.          |
| `contourpy`  | 1.3.3   | Contour calculation used by Matplotlib.                                     |
| `cycler`     | 0.12.1  | Managing repeated plot style cycles in Matplotlib.                          |
| `fonttools`  | 4.64.0  | Font handling and conversion for generated plots.                           |
| `kiwisolver` | 1.5.1   | Constraint solving used by Matplotlib's layout engine.                      |
| `pyparsing`  | 3.3.2   | Parsing structured expressions used by plotting and configuration packages. |
| `narwhals`   | 2.25.0  | Compatibility layer used by modern dataframe and Plotly integrations.       |

### Supporting dependencies

| Library           | Version     | Purpose                                                                          |
| ----------------- | ----------- | -------------------------------------------------------------------------------- |
| `cloudpickle`     | 3.1.2       | Serializing Python objects, including functions and model-related objects.       |
| `joblib`          | 1.6.0       | Efficient persistence and parallel computation for scikit-learn workflows.       |
| `packaging`       | 26.3        | Parsing and comparing package versions.                                          |
| `python-dateutil` | 2.9.0.post0 | Flexible date parsing and date arithmetic used by pandas.                        |
| `scipy`           | 1.17.1      | Scientific computing routines used by machine learning and statistics libraries. |
| `six`             | 1.17.0      | Compatibility utilities shared by Python packages.                               |
| `threadpoolctl`   | 3.6.0       | Controls threads used by native numerical libraries.                             |
| `tzdata`          | 2026.3      | Time-zone database used for timezone-aware date handling.                        |

## Setup

The project requires Python 3.11 or newer.

### Using `uv`

```
uv sync
```

### Using `pip`

```
python -m venv .venv
```

Activate the virtual environment, then install the pinned dependencies:

```
python -m pip install -r requirements.txt
```

Open the notebooks in Jupyter or the VS Code Jupyter extension after the
environment is installed. A Jupyter frontend is not included in the pinned
`requirements.txt` file.

## Workflow

1. Start with `01_eda.ipynb` to inspect distributions, missing values, and
   relationships in the survey data.
2. Use `02_preprocessing.ipynb` to clean the data and prepare model
   features.
3. Run `03_baseline_model.ipynb` to establish a reproducible benchmark.
4. Develop and compare candidate models in `04_modelling.ipynb`.
5. Use `05_submission.ipynb` to produce six probability columns in the
   required class order and prepare an early competition submission.
6. Run `09_refined_model.ipynb` for the CatBoost pipeline — in-fold target
   encoding, frequency encoding, and a tuned CatBoost model with early
   stopping and probability calibration (see "Modeling approach" above).
7. Run `10_optimized_model.ipynb` for the current best-performing
   pipeline — smoothed target encoding, leak-free frequency encoding,
   richer features, multi-seed CatBoost bagging, and an OOF-optimized
   blend with LightGBM (see "Modeling approach" above).

When evaluating models locally, use log loss and preserve the same class
mapping and probability-column order used for the final submission.

## Results

| Notebook                   | Model                                                               | OOF Log Loss                     |
| -------------------------- | ------------------------------------------------------------------- | -------------------------------- |
| `03_baseline_model.ipynb`  | _(fill in)_                                                         | _(fill in)_                      |
| `04_modelling.ipynb`       | _(fill in)_                                                         | _(fill in)_                      |
| `09_refined_model.ipynb`   | CatBoost, 5-fold CV, target + frequency encoding                    | _(fill in from your latest run)_ |
| `10_optimized_model.ipynb` | CatBoost (3-seed bagged) + LightGBM blend, smoothed target encoding | _(fill in from your latest run)_ |

Keeping this table current makes it easy to see at a glance whether a new
notebook actually improved on the last one — worth updating each time you
re-run `09_refined_model.ipynb` with different hyperparameters or features.

## About

A machine learning model that classifies the range of expenditure a tourist
spends whilst visiting Tanzania.
