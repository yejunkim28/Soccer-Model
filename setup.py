"""
Setup script to create essential directory structure for the project.
Run this script once after cloning the repository.
"""

from pathlib import Path
from tabnanny import check


def create_project_structure():
    """Create all necessary directories for the project."""

    # Define project root
    project_root = Path(__file__).parent

    # Define all directories needed
    directories = [
        # Model 1
        project_root / "model_1",

        # Model 1 data
        project_root / "model_1" / "data",
        project_root / "model_1" / "data" / "raw",
        project_root / "model_1" / "data" / "interim",
        project_root / "model_1" / "data" / "processed",

        # Model 1 models
        project_root / "model_1" / "models",
        project_root / "model_1" / "models"/ "main",
        project_root / "model_1" / "models"/ "main" / "artifacts",
        project_root / "model_1" / "models"/ "main" / "checkpoint",
        project_root / "model_1" / "models"/ "main" / "metrics",

        project_root / "model_1" / "models"/ "preprocessing",
        project_root / "model_1" / "models"/"preprocessing" / "artifacts",
        project_root / "model_1" / "models"/"preprocessing" / "metrics",



        # Model 2
        project_root / "model_2",

        # Model 2 data
        project_root / "model_2" / "data",
        project_root / "model_2" / "data" / "raw",
        
        project_root / "model_2" / "data" / "raw"  / "raw_sofifa",
        project_root / "model_2" / "data" / "raw" / "raw_sofifa" / "yearly",
        
        project_root / "model_2" / "data" / "raw" / "raw_fbref",
        project_root / "model_2" / "data" / "raw" / "raw_fbref" / "yearly",
        

        project_root / "model_2" / "data" / "interim",
        project_root / "model_2" / "data" / "processed",

        # Model 2 models
        project_root / "model_2" / "models",
        project_root / "model_2" / "models"/"main",
        project_root / "model_2" / "models"/"main" / "artifacts",
        project_root / "model_2" / "models"/"main" / "checkpoint",
        project_root / "model_2" / "models"/"main" / "metrics",

        project_root / "model_2" / "models"/"preprocessing",
        project_root / "model_2" / "models"/"preprocessing" / "artifacts",
        project_root / "model_2" / "models"/"preprocessing" / "metrics",

        # Outputs
        project_root / "outputs",
        project_root / "outputs" / "predictions",
        project_root / "outputs" / "evaluation",
        project_root / "outputs" / "evaluation" / "model_1",
        project_root / "outputs" / "evaluation" / "model_2"
        ]


    print("Creating project directory structure...")

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory.relative_to(project_root)}")

    print("\n Project setup complete")
    print("All directories have been created successfully.")


if __name__ == "__main__":
    create_project_structure()