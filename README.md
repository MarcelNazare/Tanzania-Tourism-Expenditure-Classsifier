# Tanzania Tourism Expenditure Classifier

## Project overview

This project builds a machine learning model that classifies the range of
expenditure a tourist spends while visiting Tanzania.

The dataset contains 24,675 survey responses collected by Tanzania's National
Bureau of Statistics. The responses include visitor age group, purpose of
visit (such as business, leisure, volunteering, or visiting friends and
family), and other travel characteristics.

The model predicts one of six expenditure bands:

- Low Cost
- Lower Cost
- Normal Cost
- High Cost
- Higher Cost
- Highest Cost

Predictions are submitted as class probabilities and evaluated with **log
loss**, so producing well-calibrated probabilities is important in addition to
selecting the most likely class.

Tourism contributes approximately 17% of Tanzania's GDP. A useful model could
help tour operators and the Tanzania Tourism Board provide visitors with
up-front expenditure estimates.

This challenge is hosted by **Briisp Academy**, part of the AI4D Africa
Anglophone Multidisciplinary Research Lab. It is a university hackathon open
to English-speaking African countries and runs from **September 9 to 12,
2026**.

## Project structure

```text
.
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_preprocessing.ipynb    # Data cleaning and feature preparation
│   ├── 03_baseline_model.ipynb   # Initial benchmark model
│   ├── 04_modelling.ipynb        # Model training and improvement
│   └── 05_submission.ipynb       # Prediction and submission preparation
├── .gitignore                    # Files excluded from version control
├── .python-version               # Project Python version
├── pyproject.toml                # Project metadata and direct dependencies
├── requirements.txt              # Reproducible pinned environment
├── README.md                     # Project documentation
└── uv.lock                       # Locked dependency resolution
```

The notebooks are intended to be run in order: understand the data, prepare
features, establish a baseline, train candidate models, and generate the
final submission.

## Libraries

`requirements.txt` contains the pinned runtime environment. The packages
listed below include the project's main libraries and their supporting
dependencies.

### Main machine learning and data libraries

| Library | Version | Purpose |
| --- | ---: | --- |
| `catboost` | 1.2.10 | Gradient boosting for classification, with strong support for categorical features. |
| `lightgbm` | 4.7.0 | Fast, memory-efficient gradient boosting for tabular data. |
| `numpy` | 2.4.6 | Numerical arrays and vectorized mathematical operations. |
| `pandas` | 3.0.5 | Loading, cleaning, transforming, and analyzing tabular survey data. |
| `scikit-learn` | 1.9.0 | Preprocessing, train/validation utilities, metrics, and machine learning workflows. |

### Visualization and notebook support

| Library | Version | Purpose |
| --- | ---: | --- |
| `matplotlib` | 3.11.1 | General-purpose plotting for exploratory analysis. |
| `plotly` | 7.0.0 | Interactive charts and visual data exploration. |
| `graphviz` | 0.21 | Python interface for rendering graph and tree diagrams. |
| `pillow` | 12.3.0 | Image loading and processing used by visualizations and notebooks. |
| `contourpy` | 1.3.3 | Contour calculation used by Matplotlib. |
| `cycler` | 0.12.1 | Managing repeated plot style cycles in Matplotlib. |
| `fonttools` | 4.64.0 | Font handling and conversion for generated plots. |
| `kiwisolver` | 1.5.1 | Constraint solving used by Matplotlib's layout engine. |
| `pyparsing` | 3.3.2 | Parsing structured expressions used by plotting and configuration packages. |
| `narwhals` | 2.25.0 | Compatibility layer used by modern dataframe and Plotly integrations. |

### Supporting dependencies

| Library | Version | Purpose |
| --- | ---: | --- |
| `cloudpickle` | 3.1.2 | Serializing Python objects, including functions and model-related objects. |
| `joblib` | 1.6.0 | Efficient persistence and parallel computation for scikit-learn workflows. |
| `packaging` | 26.3 | Parsing and comparing package versions. |
| `python-dateutil` | 2.9.0.post0 | Flexible date parsing and date arithmetic used by pandas. |
| `scipy` | 1.17.1 | Scientific computing routines used by machine learning and statistics libraries. |
| `six` | 1.17.0 | Compatibility utilities shared by Python packages. |
| `threadpoolctl` | 3.6.0 | Controls threads used by native numerical libraries. |
| `tzdata` | 2026.3 | Time-zone database used for timezone-aware date handling. |

## Setup

The project requires Python 3.11 or newer.

### Using `uv`

```bash
uv sync
```

### Using `pip`

```bash
python -m venv .venv
```

Activate the virtual environment, then install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Open the notebooks in Jupyter or the VS Code Jupyter extension after the
environment is installed. A Jupyter frontend is not included in the pinned
`requirements.txt` file.

## Workflow

1. Start with `01_eda.ipynb` to inspect distributions, missing values, and
   relationships in the survey data.
2. Use `02_preprocessing.ipynb` to clean the data and prepare model features.
3. Run `03_baseline_model.ipynb` to establish a reproducible benchmark.
4. Develop and compare candidate models in `04_modelling.ipynb`.
5. Use `05_submission.ipynb` to produce six probability columns in the
   required class order and prepare the competition submission.

When evaluating models locally, use log loss and preserve the same class
mapping and probability-column order used for the final submission.
