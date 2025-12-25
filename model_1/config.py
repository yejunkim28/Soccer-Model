"""
Model 1 Configuration

This config defines all paths and settings specific to Model 1.
Paths are defined relative to the model_1 directory for modularity.
"""

from pathlib import Path

# ============================================================================
# MODEL ROOT
# ============================================================================
MODEL_ROOT = Path(__file__).parent

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_DIR = MODEL_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_YEARLY_DIR = RAW_DIR / "yearly"
RAW_TOTAL_DIR = RAW_DIR / "total_raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# ============================================================================
# MODEL PATHS
# ============================================================================
MODELS_DIR = MODEL_ROOT / "models"

# Main model artifacts
MAIN_DIR = MODELS_DIR / "main"
MAIN_ARTIFACTS_DIR = MAIN_DIR / "artifacts"
MAIN_CHECKPOINT_DIR = MAIN_DIR / "checkpoint"
MAIN_METRICS_DIR = MAIN_DIR / "metrics"

# Preprocessing model artifacts
PREPROCESSING_DIR = MODELS_DIR / "preprocessing"
PREPROCESSING_ARTIFACTS_DIR = PREPROCESSING_DIR / "artifacts"
PREPROCESSING_METRICS_DIR = PREPROCESSING_DIR / "metrics"

# ============================================================================
# SHARED OUTPUT PATHS (at project root level)
# ============================================================================
PROJECT_ROOT = MODEL_ROOT.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation" / "model_1"

# ============================================================================
# SOURCE CODE PATHS
# ============================================================================
SRC_DIR = MODEL_ROOT / "src"
DATA_COLLECTION_DIR = SRC_DIR / "01_data_collection"
PREPROCESSING_SRC_DIR = SRC_DIR / "02_preprocessing"
TRAINING_DIR = SRC_DIR / "03_training"
INFERENCE_DIR = SRC_DIR / "04_inference"

# ============================================================================
# SCRIPTS PATHS
# ============================================================================
SCRIPTS_DIR = MODEL_ROOT / "scripts"

# ============================================================================
# NOTEBOOKS PATHS
# ============================================================================
NOTEBOOKS_DIR = MODEL_ROOT / "notebooks"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_paths() -> dict:
    """
    Get all paths for Model 1 as a dictionary.
    
    Returns:
        Dictionary containing all path configurations
    """
    return {
        "root": MODEL_ROOT,
        "data": {
            "raw": RAW_DIR,
            "raw_yearly": RAW_YEARLY_DIR,
            "raw_total": RAW_TOTAL_DIR,
            "interim": INTERIM_DIR,
            "processed": PROCESSED_DIR,
        },
        "models": {
            "main": {
                "artifacts": MAIN_ARTIFACTS_DIR,
                "checkpoint": MAIN_CHECKPOINT_DIR,
                "metrics": MAIN_METRICS_DIR,
            },
            "preprocessing": {
                "artifacts": PREPROCESSING_ARTIFACTS_DIR,
                "metrics": PREPROCESSING_METRICS_DIR,
            },
        },
        "outputs": {
            "predictions": PREDICTIONS_DIR,
            "evaluation": EVALUATION_DIR,
        },
        "src": {
            "data_collection": DATA_COLLECTION_DIR,
            "preprocessing": PREPROCESSING_SRC_DIR,
            "training": TRAINING_DIR,
            "inference": INFERENCE_DIR,
        },
        "scripts": SCRIPTS_DIR,
        "notebooks": NOTEBOOKS_DIR,
    }


def ensure_directories():
    """Create all directories if they don't exist."""
    paths = get_all_paths()
    
    # Data directories
    for path in paths["data"].values():
        path.mkdir(parents=True, exist_ok=True)
    
    # Model directories
    for model_type in paths["models"].values():
        for path in model_type.values():
            path.mkdir(parents=True, exist_ok=True)
    
    # Output directories
    for path in paths["outputs"].values():
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Verify paths when run directly
    ensure_directories()
    print("Model 1 directory structure verified/created:")
    for key, value in get_all_paths().items():
        print(f"\n{key.upper()}:")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    print(f"  {sub_key}:")
                    for item_key, item_value in sub_value.items():
                        print(f"    {item_key}: {item_value}")
                else:
                    print(f"  {sub_key}: {sub_value}")
        else:
            print(f"  {value}")
