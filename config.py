"""
Configuration file for all project paths.
Run setup.py first to create these directories.

This config matches the model-separated structure defined in setup.py
where model_1 and model_2 are independent projects.
"""

from pathlib import Path

# ============================================================================
# PROJECT ROOT
# ============================================================================
PROJECT_ROOT = Path(__file__).parent

# ============================================================================
# MODEL 1 PATHS
# ============================================================================
MODEL_1_ROOT = PROJECT_ROOT / "model_1"

# Model 1 - Data
MODEL_1_DATA_DIR = MODEL_1_ROOT / "data"
MODEL_1_RAW_DIR = MODEL_1_DATA_DIR / "raw"
MODEL_1_INTERIM_DIR = MODEL_1_DATA_DIR / "interim"
MODEL_1_PROCESSED_DIR = MODEL_1_DATA_DIR / "processed"

# Model 1 - Models
MODEL_1_MODELS_DIR = MODEL_1_ROOT / "models"
MODEL_1_MAIN_DIR = MODEL_1_MODELS_DIR / "main"
MODEL_1_MAIN_ARTIFACTS_DIR = MODEL_1_MAIN_DIR / "artifacts"
MODEL_1_MAIN_CHECKPOINT_DIR = MODEL_1_MAIN_DIR / "checkpoint"
MODEL_1_MAIN_METRICS_DIR = MODEL_1_MAIN_DIR / "metrics"

MODEL_1_PREPROCESSING_DIR = MODEL_1_MODELS_DIR / "preprocessing"
MODEL_1_PREPROCESSING_ARTIFACTS_DIR = MODEL_1_PREPROCESSING_DIR / "artifacts"
MODEL_1_PREPROCESSING_METRICS_DIR = MODEL_1_PREPROCESSING_DIR / "metrics"

# ============================================================================
# MODEL 2 PATHS
# ============================================================================
MODEL_2_ROOT = PROJECT_ROOT / "model_2"

# Model 2 - Data
MODEL_2_DATA_DIR = MODEL_2_ROOT / "data"
MODEL_2_RAW_DIR = MODEL_2_DATA_DIR / "raw"
MODEL_2_RAW_YEARLY_DIR = MODEL_2_RAW_DIR / "yearly"
MODEL_2_RAW_TOTAL_DIR = MODEL_2_RAW_DIR / "total_raw"
MODEL_2_INTERIM_DIR = MODEL_2_DATA_DIR / "interim"
MODEL_2_PROCESSED_DIR = MODEL_2_DATA_DIR / "processed"

# Model 2 - Models
MODEL_2_MODELS_DIR = MODEL_2_ROOT / "models"
MODEL_2_MAIN_DIR = MODEL_2_MODELS_DIR / "main"
MODEL_2_MAIN_ARTIFACTS_DIR = MODEL_2_MAIN_DIR / "artifacts"
MODEL_2_MAIN_CHECKPOINT_DIR = MODEL_2_MAIN_DIR / "checkpoint"
MODEL_2_MAIN_METRICS_DIR = MODEL_2_MAIN_DIR / "metrics"

MODEL_2_PREPROCESSING_DIR = MODEL_2_MODELS_DIR / "preprocessing"
MODEL_2_PREPROCESSING_ARTIFACTS_DIR = MODEL_2_PREPROCESSING_DIR / "artifacts"
MODEL_2_PREPROCESSING_METRICS_DIR = MODEL_2_PREPROCESSING_DIR / "metrics"

# ============================================================================
# OUTPUTS (Shared)
# ============================================================================
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
EVALUATION_MODEL_1_DIR = EVALUATION_DIR / "model_1"
EVALUATION_MODEL_2_DIR = EVALUATION_DIR / "model_2"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_config(model_name: str) -> dict:
    """
    Get all paths for a specific model.
    
    Args:
        model_name: Either 'model_1' or 'model_2'
    
    Returns:
        Dictionary containing all relevant paths for the model
    """
    if model_name == "model_1":
        return {
            "root": MODEL_1_ROOT,
            "data": {
                "raw": MODEL_1_RAW_DIR,
                "interim": MODEL_1_INTERIM_DIR,
                "processed": MODEL_1_PROCESSED_DIR,
            },
            "models": {
                "main": {
                    "artifacts": MODEL_1_MAIN_ARTIFACTS_DIR,
                    "checkpoint": MODEL_1_MAIN_CHECKPOINT_DIR,
                    "metrics": MODEL_1_MAIN_METRICS_DIR,
                },
                "preprocessing": {
                    "artifacts": MODEL_1_PREPROCESSING_ARTIFACTS_DIR,
                    "metrics": MODEL_1_PREPROCESSING_METRICS_DIR,
                },
            },
            "outputs": {
                "predictions": PREDICTIONS_DIR,
                "evaluation": EVALUATION_MODEL_1_DIR,
            },
        }
    elif model_name == "model_2":
        return {
            "root": MODEL_2_ROOT,
            "data": {
                "raw": MODEL_2_RAW_DIR,
                "raw_yearly": MODEL_2_RAW_YEARLY_DIR,
                "raw_total": MODEL_2_RAW_TOTAL_DIR,
                "interim": MODEL_2_INTERIM_DIR,
                "processed": MODEL_2_PROCESSED_DIR,
            },
            "models": {
                "main": {
                    "artifacts": MODEL_2_MAIN_ARTIFACTS_DIR,
                    "checkpoint": MODEL_2_MAIN_CHECKPOINT_DIR,
                    "metrics": MODEL_2_MAIN_METRICS_DIR,
                },
                "preprocessing": {
                    "artifacts": MODEL_2_PREPROCESSING_ARTIFACTS_DIR,
                    "metrics": MODEL_2_PREPROCESSING_METRICS_DIR,
                },
            },
            "outputs": {
                "predictions": PREDICTIONS_DIR,
                "evaluation": EVALUATION_MODEL_2_DIR,
            },
        }
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'model_1' or 'model_2'.")