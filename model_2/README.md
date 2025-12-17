# Model 2

Model 2 description and documentation.

## Structure

```
model_2/
├── data/           # Model 2 specific data
├── models/         # Trained models and metrics
├── notebooks/      # Exploratory notebooks
├── outputs/        # Predictions and reports
├── scripts/        # Executable scripts
└── src/            # Source code
    ├── data/           # Data collection & validation
    ├── features/       # Feature engineering
    ├── preprocessing/  # Data preprocessing
    ├── models/         # Model definitions
    ├── training/       # Training logic
    └── inference/      # Prediction logic
```

## Usage

```python
# Example workflow
from model_2.src.data.collection import load_yearly_data
from model_2.src.preprocessing.preprocess import Model2Preprocessor
from model_2.src.models.model import Model2
from model_2.src.training.trainer import Model2Trainer

# Load data
df = load_yearly_data(2024)

# Preprocess
preprocessor = Model2Preprocessor()
df_processed = preprocessor.fit_transform(df)

# Train
trainer = Model2Trainer(config={})
trainer.train()

# Predict
# ... inference logic
```
