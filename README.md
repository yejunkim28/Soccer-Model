# Soccer Player Performance Prediction System

A research-grade system for predicting **next-season football (soccer) player performance** using longitudinal match statistics from **1995–2024**.  
The project emphasizes **temporal modeling**, **multi-output regression**, and **reproducible inference**, rather than point-estimate heuristics.

The system is built to answer one core question:

> Given a player’s historical performance trajectory, what can we reasonably expect in the _next_ season?

---

## Project Goals

- Model player performance as a **time series**, not isolated seasons
- Support **variable career lengths** via adaptive rolling windows
- Predict **multiple performance metrics simultaneously**
- Separate **data, models, logic, and outputs** cleanly
- Enable safe iteration without breaking downstream components

This is not a dashboard or demo—it is an extensible modeling pipeline.

---

## High-Level Architecture

The system is divided into four strictly separated stages:

1. **Data Collection**  
   Raw season-level data ingestion and normalization

2. **Preprocessing**  
   Feature engineering and learned missing-value estimation

3. **Model Training & Evaluation**  
   Rolling-window, multi-output regression models with stored metrics

4. **Inference**  
   Automated next-season prediction using the best-valid model per player

Each stage can be modified or retrained independently.

---

## Directory Structure

```bash
Coding_Project/
├── data/
│ ├── years/ # Raw season-level CSVs (1995–2024)
│ ├── fbref_total_fielders/ # Aggregated raw data
│ └── processed_whole/ # Final merged dataset
│
├── models/
│ ├── main/
│ │ ├── model_1/
│ │ │ ├── artifacts/ # Trained prediction models (per window)
│ │ │ └── metrics/ # Evaluation metrics (per target, per window)
│ │ └── model_2/
│ │
│ └── preprocessing/
│ ├── model_1/ # Models for filling missing values
│ └── model_2/
│
├── src/
│ ├── preprocessing/
│ │ ├── data_collection.py
│ │ └── preprocess.py
│ │
│ ├── training/
│ │ ├── train_model_1.py
│ │ ├── train_model_2.py
│ │ └── evaluate.py
│ │
│ └── inference/
│ └── model_runner.py
│
├── output/ # Prediction outputs
├── notebooks/ # Exploration/debugging only
├── variables.py # Shared constants
├── requirements.txt
└── README.md
```

**Design rule**

- `src/` → executable logic
- `models/` → trained artifacts and metrics
- `data/` → immutable inputs
- `output/` → disposable results

---

## Data Flow

### Raw → Processed

- Season CSVs (e.g. `9596`, `9697`) are normalized to calendar years (`1995`, `1996`, …)
- Player records are sorted strictly by season
- Players with insufficient historical data are filtered early
- Output is a single longitudinal dataset in `data/processed_whole/`

---

## Missing-Value Strategy

Missing values are **not** filled using simple heuristics.

Instead:

- A **multi-output XGBoost regression model** is trained on players with complete data
- Only columns with demonstrable predictive signal are estimated
- Filled values are treated as _model-derived estimates_, not ground truth

Preprocessing models and scalers are stored under: "models/preprocessing/"

---

## Modeling Strategy

### Rolling-Window Training

Multiple models are trained using different historical windows:

| Window | Interpretation          |
| ------ | ----------------------- |
| 3      | Short-term form         |
| 5–7    | Medium-term development |
| 10     | Long-term career trend  |

Each model:

- Uses the previous `W` seasons as features
- Predicts **all target statistics simultaneously**
- Is trained independently to avoid temporal leakage

This avoids bias against younger players while still leveraging long careers.

---

## Target Variables

Current targets focus on **advanced per-90 and progression metrics**, including:

- Shooting efficiency
- Expected goals/assists
- Passing and progression indicators
- Defensive and recovery actions

**Known limitation**  
These metrics are not intuitive for non-technical audiences.

**Planned direction**

- Train a comprehensive internal model
- Post-process outputs into:
  - Basic stats (e.g. goals, assists)
  - Intermediate metrics
  - Advanced analytics

---

## Training Pipeline

1. Construct rolling-window samples
2. Train a multi-output regression model per window size
3. Evaluate performance per target using MAE, RMSE, and R²
4. Persist:
   - Model artifacts (`.pkl`)
   - Metrics (`.csv`)

Artifacts are versioned by window size and stored immutably.

---

## Inference Pipeline

Inference predicts the _next_ season for players active in the latest year:

- Only players appearing in the most recent season are considered
- For each player:
  - Determine the largest valid window given career length
  - Load the corresponding trained model
  - Generate next-season predictions
- Outputs are consolidated into a single DataFrame and saved to `output/`

This mirrors real-world forecasting constraints.

---

## Configuration & Reproducibility

- All constants are centralized in `variables.py`
- No implicit globals or hidden state
- Models are deterministic given the same data and configuration
- Retraining requires explicit script execution

---

## Usage Overview

Typical workflow:

1. Update or add raw data in `data/`
2. Run preprocessing scripts
3. Train or retrain models as needed
4. Run inference to generate next-season predictions

Notebooks are for exploration only and are not part of the production pipeline.

---

## Philosophy

This system prioritizes:

- Temporal correctness over convenience
- Explicit modeling decisions over black-box automation
- Clear separation of concerns

If a result cannot be explained or reproduced, it is considered incorrect.
