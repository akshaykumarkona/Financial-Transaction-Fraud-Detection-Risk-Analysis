# Financial Transaction Fraud Detection & Risk Analysis

## Project overview

Financial fraud is difficult to identify because fraudulent transactions are rare compared with legitimate activity. A model that predicts every transaction as legitimate can appear highly accurate while failing to detect fraud. This project develops a transaction-level fraud detection workflow using exploratory data analysis (EDA), behavioral and balance-based feature engineering, class-imbalance handling, and a Random Forest classifier.

The goal is to distinguish suspicious transactions from normal activity while emphasizing **recall, precision, and F1-score** rather than accuracy alone. The analysis also investigates transaction patterns and account-balance behavior that may be useful for risk screening.

### At a glance

| Item | Details |
|---|---|
| Problem | Binary classification: fraudulent vs. legitimate transactions |
| Dataset analyzed | 1,048,575 transactions, 11 original columns |
| Fraud cases | 1,142 (0.109%) |
| Legitimate cases | 1,047,433 (99.891%) |
| Algorithm | Random Forest classifier |
| Imbalance treatment | SMOTETomek on training data |
| Evaluation | 25% held-out test split; accuracy, precision, recall, F1, confusion matrix |
| Tools | Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, imbalanced-learn, SciPy |

## Business problem and motivation

Fraud monitoring teams must balance two competing costs:

- **Missed fraud (false negatives):** suspicious transactions go undetected, potentially increasing financial losses.
- **False alarms (false positives):** legitimate transactions are flagged, increasing investigation workload and customer friction.

This project treats fraud detection as a highly imbalanced classification problem and prioritizes detecting fraud without ignoring the operational burden of false positives. It is a research/prototyping workflow, **not a production-ready payment-blocking system**.

## Dataset

The analysis uses a transaction dataset loaded from `Fraud.csv`. The analyzed subset contains **1,048,575 records** and the following 11 original fields:

| Feature | Description |
|---|---|
| `step` | Simulation time-step index (hourly) |
| `type` | Transaction category: `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, or `TRANSFER` |
| `amount` | Transaction amount |
| `nameOrig` | Originating account identifier |
| `oldbalanceOrg` | Originating account balance before transaction |
| `newbalanceOrig` | Originating account balance after transaction |
| `nameDest` | Destination account identifier |
| `oldbalanceDest` | Destination balance before transaction |
| `newbalanceDest` | Destination balance after transaction |
| `isFraud` | Ground-truth label: `1` = fraud, `0` = legitimate |
| `isFlaggedFraud` | Existing rule-based flag; constant at `0` in the analyzed data |

**Data-quality checks:** The analysis reported no missing values and no duplicate rows. Since `isFlaggedFraud` had only one unique value, it was removed from subsequent analysis. An auxiliary row-index field was also created during processing; this is not an original dataset feature.

### Class imbalance

| Class | Count | Share |
|---|---:|---:|
| Legitimate (`0`) | 1,047,433 | 99.891% |
| Fraudulent (`1`) | 1,142 | 0.109% |

Only about **1 in 918 transactions** was fraudulent. This makes overall accuracy a poor standalone indicator of fraud-detection quality.

## Methodology

### 1. Exploratory data analysis

The investigation examines:

- Fraud versus non-fraud class frequencies and proportions.
- Transaction-type frequencies and the distribution of transaction time steps.
- Transaction amounts and originating/destination account balances.
- Relationships between numerical variables and the fraud label.
- Account and destination behavior, including transaction frequency and balance inconsistencies.
- Statistical comparisons between fraud and non-fraud groups using techniques available in SciPy.

The five transaction categories in the analyzed data have these counts:

| Type | Transactions |
|---|---:|
| `CASH_OUT` | 373,641 |
| `PAYMENT` | 353,873 |
| `CASH_IN` | 227,130 |
| `TRANSFER` | 86,753 |
| `DEBIT` | 7,178 |

### 2. Feature engineering

The workflow creates features intended to capture suspicious monetary behavior, balance changes, and account activity:

| Engineered feature | Purpose |
|---|---|
| `amount_ratio` | Transaction amount relative to the originating balance (`amount / (oldbalanceOrg + 1)`) |
| `balance_diff_orig` | Difference between originating balance before and after the transaction |
| `balance_diff_dest` | Difference between destination balance after and before the transaction |
| `origin_balance_error` | Absolute discrepancy between expected and observed originating balance |
| `dest_balance_error` | Absolute discrepancy between expected and observed destination balance |
| `is_zero_balance_after` | Indicates whether the originating account has a zero balance afterward |
| `is_merchant_dest`, `is_merchant_orig` | Indicators derived from account identifier prefixes |
| `orig_txn_count`, `dest_txn_count` | Number of transactions associated with an account |
| `orig_avg_amount`, `dest_avg_amount` | Mean transaction amount by originating/destination account |
| `amount_deviation_orig` | Deviation from the originating account's mean transaction amount |

Categorical transaction types are encoded for model input. A destination-account category is also derived and encoded.

### 3. Train/test split and class balancing

The data is split into **75% training** and **25% testing** using a fixed random seed (`random_state=14`).

| Partition | Records |
|---|---:|
| Training | 786,431 |
| Testing | 262,144 |

The training partition initially contains **785,581 legitimate** and **850 fraudulent** records. **SMOTETomek** is applied to the training data to combine synthetic minority-class oversampling with Tomek-link cleaning. The resulting resampled training set has **783,508 observations per class** (1,567,016 total). The held-out test data is not resampled.

### 4. Model training and evaluation

A `RandomForestClassifier` is trained on the resampled training data. Predictions are evaluated on the held-out test partition using multiple classification metrics and a confusion matrix. The analysis also reports 10-fold cross-validation on the resampled training set.

## Results

### Held-out test-set performance

| Metric | Reported result | Interpretation |
|---|---:|---|
| **Accuracy** | **99.98%** | Overall fraction of correct predictions; less informative under severe imbalance |
| **Precision** | **0.83** | About 83% of predicted fraud cases were truly fraudulent |
| **Recall** | **0.98** | About 98% of actual fraud cases were detected |
| **F1-score** | **0.90** | Harmonic mean of precision and recall |

### Confusion matrix

| Actual / Predicted | Legitimate (`0`) | Fraud (`1`) |
|---|---:|---:|
| **Legitimate (`0`)** | 261,792 (TN) | 60 (FP) |
| **Fraud (`1`)** | 5 (FN) | 287 (TP) |

**What this means:** In the reported held-out evaluation, the classifier detected **287 of 292 fraud cases**, missed **5**, and incorrectly flagged **60 legitimate transactions**. These numbers illustrate the trade-off between catching fraud and creating false-positive investigations.

The workflow also reports approximately **1.00 mean F1-score in 10-fold cross-validation on the resampled training data**. This cross-validation result should **not** be interpreted as independent evidence of near-perfect generalization: resampling before cross-validation can cause synthetic-sample overlap between folds and inflate scores. The held-out test results are more informative, though additional validation is needed (see below).

## Interpretation and limitations

This project demonstrates a complete experimental fraud-classification workflow and shows promising **reported** held-out results. However, the metrics are **not yet a reliable production-performance estimate** without addressing several methodological risks:

1. **Feature leakage:** Some account-level aggregate features were calculated using the full dataset before splitting. They should be learned only from training/history data and applied to later observations without access to future transactions.
2. **Row-index leakage:** The experimental feature set includes an auxiliary `index_column`, which can encode ordering information and should be excluded from a defensible final model.
3. **Post-transaction information:** Features involving updated account balances may only be available after transaction execution. Their suitability depends on whether the model is intended for pre-authorization screening or post-transaction investigation.
4. **Cross-validation leakage:** SMOTETomek was applied before the reported 10-fold cross-validation. A safer evaluation places resampling **inside each training fold** using an `imblearn` pipeline.
5. **Encoding consistency:** Categorical encoders should be fitted on training data only and reused unchanged on validation/test data.
6. **Temporal generalization:** A chronological holdout and out-of-time validation are important because transaction patterns can drift over time.
7. **Operational metrics:** Precision–recall curves, PR-AUC, recall at a fixed false-positive rate, threshold tuning, and investigation costs would help evaluate real-world usefulness.

These are improvement opportunities, not claims that the reported scores have already been corrected or revalidated.

## Recommended reproducible workflow

For a stronger end-to-end implementation:

```text
Raw transactions
      |
      v
Data validation and exploratory analysis
      |
      v
Chronological train / validation / test split
      |
      v
Leakage-safe feature engineering
      |
      v
Train-only categorical preprocessing
      |
      v
SMOTETomek within training / CV folds
      |
      v
Random Forest training and tuning
      |
      v
Out-of-time evaluation + threshold selection
      |
      v
Fraud risk scores and investigation prioritization
```

## Technology stack

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn (`RandomForestClassifier`, train/test split, cross-validation, classification metrics)
- **Imbalanced learning:** imbalanced-learn (`SMOTETomek`)
- **Statistical analysis:** SciPy

## Getting started

The project uses a local CSV dataset named `Fraud.csv`. To reproduce the experiment, obtain an appropriately licensed copy of the dataset and place it in your project directory. The dataset is **not included or linked here**.

Install the core dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

python -m pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn
```

Optional dependency for exporting DataFrame images:

```bash
python -m pip install dataframe-image
```

Run the project's Python analysis code in your preferred local Python environment, with the CSV available at the path configured in the code. **No standalone command-line application, API, or hosted demo is claimed in this repository description.**

## Future development

- Rebuild preprocessing and resampling as a leakage-safe, reproducible pipeline.
- Benchmark Logistic Regression, gradient-boosted trees, and alternative imbalance strategies against Random Forest.
- Compare results using chronological splits and multiple random seeds.
- Add model explainability with feature importance and SHAP, while checking for unstable or leaked signals.
- Calibrate risk scores and select thresholds based on fraud-loss and review-capacity assumptions.
- Monitor data drift, false-positive rates, and fraud recall over time.
- Package the validated model as a batch-scoring pipeline or API with a lightweight monitoring dashboard.

## Key takeaway

**Fraud detection is not simply an accuracy-maximization problem.** With only **0.109%** fraud in the analyzed data, meaningful evaluation requires attention to the minority class and to the operational cost of false positives. The experiment reports **98% recall and 90% F1-score** on a held-out partition, while highlighting the importance of leakage-safe validation before real-world deployment.


