# Soccer Prediction Advanced Models

This project implements advanced machine learning models to predict various soccer performance metrics. The models included are XGBoost, LightGBM, and a Neural Network. Each model is designed to predict multiple target variables based on player statistics and other relevant features.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to provide a robust framework for predicting soccer player performance using advanced machine learning techniques. The models are trained on historical player data and can be used to forecast future performance metrics.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yejunkim28/Soccer-Model.git
cd Soccer-Model
pip install -r requirements.txt
```

## Usage

The project includes Jupyter notebooks for training and evaluating each model. You can run the notebooks to see the training process and results:

- `notebooks/xgboost_training.ipynb`: Train and evaluate the XGBoost model.
- `notebooks/lightgbm_training.ipynb`: Train and evaluate the LightGBM model.
- `notebooks/neural_network_training.ipynb`: Train and evaluate the Neural Network model.
- `notebooks/model_comparison.ipynb`: Compare the performance of all models.

## Models

### XGBoost
- Implements the XGBoost algorithm for regression tasks.
- Includes a training pipeline, parameter configuration, and methods for fitting the model and making predictions.

### LightGBM
- Implements the LightGBM algorithm for regression tasks.
- Similar structure to the XGBoost model with its own parameter settings.

### Neural Network
- Implements a neural network model for predicting target variables.
- Configurable architecture and training parameters.

## Visualization

The project includes visualization tools to analyze model performance:

- Loss curves for each model during training.
- Performance metrics plots to compare model accuracy.
- Feature importance visualizations to understand model decisions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features you'd like to add.

## License

This project is licensed under the MIT License. See the LICENSE file for details.