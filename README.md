# Soccer Player Performance Prediction System

A research-grade system for predicting **next-season football (soccer) player performance** using longitudinal match statistics from **1995–2024**.

This project contains **two independent prediction models** with different approaches, features, and target variables. Each model is completely self-contained with its own data pipeline, training logic, and inference system.

---

## Project Structure

This repository uses a **model-separated architecture** where Model 1 and Model 2 are independent projects:

```
soccer_prediction/
├── config.py                    # Centralized path configuration
├── setup.py                     # Creates directory structure
├── requirements.txt             # Python dependencies
├── variables.py                 # Shared constants (legacy)
│
├── model_1/                     # Model 1 - Complete pipeline
│   ├── data/
│   │   ├── raw/                 # Model 1 raw data
│   │   ├── interim/             # Intermediate processing
│   │   └── processed/           # Final features
│   ├── models/
│   │   ├── main/
│   │   │   ├── artifacts/       # Trained models
│   │   │   ├── checkpoint/      # Training checkpoints
│   │   │   └── metrics/         # Performance metrics
│   │   └── preprocessing/
│   │       ├── artifacts/       # Imputation models
│   │       └── metrics/         # Preprocessing metrics
│   ├── src/
│   │   ├── 01_data_collection/  # Data loading
│   │   ├── 02_preprocessing/    # Feature engineering
│   │   ├── 03_training/         # Model training
│   │   └── 04_inference/        # Predictions
│   └── notebooks/               # Exploratory analysis
│
├── model_2/                     # Model 2 - Complete pipeline
│   ├── data/
│   │   ├── raw/
│   │   │   ├── yearly/          # Year-by-year data
│   │   │   └── total_raw/       # Aggregated data
│   │   ├── interim/
│   │   └── processed/
│   ├── models/
│   │   ├── main/
│   │   │   ├── artifacts/
│   │   │   ├── checkpoint/
│   │   │   └── metrics/
│   │   └── preprocessing/
│   │       ├── artifacts/
│   │       └── metrics/
│   ├── src/
│   │   ├── data/                # Data collection & validation
│   │   ├── features/            # Feature engineering
│   │   ├── preprocessing/       # Data preprocessing
│   │   ├── models/              # Model definitions
│   │   ├── training/            # Training logic
│   │   └── inference/           # Prediction logic
│   ├── notebooks/
│   └── README.md                # Model 2 documentation
│
└── outputs/                     # Shared outputs directory
    ├── predictions/             # Model predictions
    └── evaluation/
        ├── model_1/             # Model 1 evaluations
        └── model_2/             # Model 2 evaluations
```

---

## Key Design Principles

### Complete Separation

- **Model 1** and **Model 2** are fully independent
- Each model has its own data, features, and logic
- No shared code between models (except `config.py`)
- Models can be developed, trained, and deployed separately

### Structured Pipelines

Each model follows a clear pipeline:

1. **Data Collection** → Load and validate raw data
2. **Preprocessing** → Clean, transform, and engineer features
3. **Training** → Train models with different configurations
4. **Inference** → Generate predictions for new data

### Centralized Configuration

- All paths managed in `config.py`
- Use `get_model_config("model_1")` or `get_model_config("model_2")`
- No hardcoded paths in source code

---

## Getting Started

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directory structure
python setup.py
```

### 2. Configuration

All paths are defined in `config.py`. The configuration provides:

- Model-specific data directories
- Model artifact locations
- Output directories
- Helper functions for path management

### 3. Working with Model 1

```bash
# Navigate to Model 1
cd model_1

# Model 1 uses numbered workflow directories
# Run scripts in order:
# 01_data_collection → 02_preprocessing → 03_training → 04_inference
```

### 4. Working with Model 2

```bash
# Navigate to Model 2
cd model_2

# Model 2 uses clean module names
# Access via: src/data/, src/features/, src/preprocessing/, etc.
```

---

## Model Comparison

| Aspect          | Model 1                            | Model 2                               |
| --------------- | ---------------------------------- | ------------------------------------- |
| **Structure**   | Numbered workflow (01*, 02*, ...)  | Named modules (data/, features/, ...) |
| **Status**      | ✅ Completed                       | 🚧 In Development                     |
| **Data Source** | Shared raw data                    | Separate yearly + total data          |
| **Features**    | Rolling windows, temporal features | TBD - Different feature set           |
| **Targets**     | Advanced per-90 metrics            | TBD - Different targets               |

---

## Common Workflows

### Training a Model

**Model 1:**

```python
# From model_1 directory
from src.training.trainer import ModelTrainer

trainer = ModelTrainer(...)
trainer.train()
```

**Model 2:**

```python
# From model_2 directory
from src.training.trainer import Model2Trainer

trainer = Model2Trainer(config={...})
trainer.train()
```

### Making Predictions

**Model 1:**

```python
from src.inference.predictor import make_predictions

predictions = make_predictions(data)
```

**Model 2:**

```python
from src.inference.predictor import Model2Predictor

predictor = Model2Predictor()
predictor.load_model(path)
predictions = predictor.predict(data)
```

---

## Development Guidelines

1. **Never mix model code** - Keep Model 1 and Model 2 completely separate
2. **Use config.py** - Always use centralized paths, never hardcode
3. **Document changes** - Update model-specific READMEs when making changes
4. **Test independently** - Each model should work without the other
5. **Version artifacts** - Save models with clear version/window identifiers

---

## Configuration & Reproducibility

- All paths centralized in `config.py`
- Shared constants in `variables.py` (legacy)
- Models are deterministic given same data and configuration
- Run `setup.py` to recreate directory structure

---

## Outputs

All predictions and evaluations are saved to the shared `outputs/` directory:

- `outputs/predictions/` - Model predictions
- `outputs/evaluation/model_1/` - Model 1 metrics
- `outputs/evaluation/model_2/` - Model 2 metrics
