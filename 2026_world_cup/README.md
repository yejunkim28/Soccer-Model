# 2026 FIFA World Cup Prediction System

Comprehensive prediction system for the 2026 FIFA World Cup using player-level predictions aggregated to team performance forecasts.

## 🎯 Project Overview

This module predicts the outcomes of the 2026 FIFA World Cup by:

1. Using existing player prediction models (model_1 & model_2)
2. Aggregating player stats into team-level strengths
3. Training match outcome prediction models
4. Simulating the entire tournament with Monte Carlo methods

## 📊 Key Features

- **Player-to-Team Aggregation**: Convert individual player predictions into team strength metrics
- **Match Prediction**: Predict win/draw/loss probabilities for any matchup
- **Tournament Simulation**: Monte Carlo simulation of entire tournament (10,000+ runs)
- **Visualization**: Interactive dashboards and bracket visualizations
- **Confidence Intervals**: Probabilistic predictions with uncertainty quantification

## 🗂️ Project Structure

```
2026_world_cup/
├── config.py                      # Configuration and paths
├── README.md                      # This file
├── requirements.txt               # Dependencies
├── WORLD_CUP_PREDICTION_PLAN.md  # Detailed implementation plan
│
├── data/                          # All data files
│   ├── raw/                       # Raw collected data
│   ├── processed/                 # Processed features
│   └── historical/                # Historical World Cup data
│
├── models/                        # Trained models
│   ├── match_predictor.pkl
│   └── goals_predictor.pkl
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_team_aggregation.ipynb
│   ├── 03_match_modeling.ipynb
│   └── 04_tournament_simulation.ipynb
│
├── scripts/                       # Executable scripts
│   ├── collect_data.py
│   ├── prepare_teams.py
│   ├── train_match_model.py
│   ├── simulate_tournament.py
│   └── generate_predictions.py
│
├── src/                           # Source code
│   ├── data/                      # Data collection and processing
│   ├── features/                  # Feature engineering
│   ├── models/                    # Model definitions
│   └── visualization/             # Visualization tools
│
└── outputs/                       # Generated predictions
    ├── predictions/               # Match and tournament predictions
    └── visualizations/            # Charts and dashboards
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize Directories

```bash
python config.py
```

### 3. Collect Data

```bash
# Collect qualified teams, rosters, and rankings
python scripts/collect_data.py
```

### 4. Prepare Team Data

```bash
# Aggregate player predictions into team strengths
python scripts/prepare_teams.py --use-model model_1
```

### 5. Train Match Prediction Model

```bash
# Train on historical World Cup data
python scripts/train_match_model.py
```

### 6. Generate Predictions

```bash
# Simulate tournament and generate predictions
python scripts/simulate_tournament.py --n-simulations 10000
```

## 📈 Methodology

### 1. Player Prediction Base Layer

- Uses pre-trained models (model_1 or model_2)
- Forecasts player performance for 2025-2026 season
- Includes ratings, goals, assists, defensive metrics

### 2. Team Aggregation Layer

- Maps players to national team rosters
- Calculates team strength metrics:
  - Overall squad rating
  - Starting XI vs bench strength
  - Position-specific strengths (attack, midfield, defense)
  - Squad balance and depth
  - Experience and age profile

### 3. Match Prediction Layer

- Gradient boosting model (XGBoost/LightGBM)
- Features:
  - Team strength differentials
  - FIFA rankings
  - Historical head-to-head
  - Home advantage
  - Tournament stage importance
- Outputs: Win/Draw/Loss probabilities

### 4. Tournament Simulation

- Monte Carlo simulation (10,000+ iterations)
- Group stage point calculations
- Knockout bracket progression
- Penalty shootout modeling
- Aggregates results into win probabilities

## 📊 Expected Outputs

1. **Match Predictions**: Probability distribution for each match
2. **Group Stage**: Qualification probabilities for all teams
3. **Knockout Bracket**: Most likely progression paths
4. **Tournament Winner**: Top 20 teams with championship odds
5. **Player Awards**: Golden Boot and Golden Ball predictions
6. **Visualizations**: Interactive brackets, probability heatmaps

## 🔧 Configuration

Key settings in `config.py`:

```python
# Tournament settings
TOURNAMENT_CONFIG = {
    "year": 2026,
    "num_teams": 48,
    "host_countries": ["USA", "Canada", "Mexico"],
    "groups": 16,
    "teams_per_group": 3
}

# Model settings
MODEL_CONFIG = {
    "match_model": {"type": "xgboost", ...},
    "simulation": {"n_simulations": 10000, ...}
}
```

## 📚 Data Requirements

### Required Data (To Collect)

- ✅ Qualified teams list (48 teams)
- ✅ Squad rosters with player names
- ✅ FIFA rankings (current + historical)
- ✅ Recent international match results (2022-2026)
- ✅ Tournament structure and schedule

### Available Data (Already Have)

- ✅ Player predictions for 2025
- ✅ Historical player statistics
- ✅ Player attributes and ratings

## 🎯 Validation Strategy

- **Historical Validation**: Train on 2006-2018, test on 2022
- **Metrics**: Accuracy, Log Loss, Brier Score, Top-4 accuracy
- **Baseline Comparison**: FIFA rankings only
- **Target**: >55% match prediction accuracy

## 📝 Development Roadmap

### Phase 1: Data Collection (Week 1-2) ⏳

- [ ] Scrape qualified teams list
- [ ] Collect squad rosters
- [ ] Download FIFA rankings
- [ ] Get recent match results

### Phase 2: Team Aggregation (Week 2-3) ⏳

- [ ] Player-team matching
- [ ] Team strength calculation
- [ ] Feature engineering

### Phase 3: Model Training (Week 3-4) ⏳

- [ ] Collect historical WC data
- [ ] Train match prediction model
- [ ] Validate on 2022 World Cup

### Phase 4: Tournament Simulation (Week 5) ⏳

- [ ] Implement Monte Carlo simulator
- [ ] Generate predictions
- [ ] Create visualizations

### Phase 5: Deployment (Week 6) ⏳

- [ ] Build interactive dashboard
- [ ] Generate final reports
- [ ] Prepare presentation

## 🛠️ Technologies

- **ML Frameworks**: scikit-learn, XGBoost, LightGBM
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: streamlit
- **Web Scraping**: beautifulsoup4, requests

## 📖 Usage Examples

### Predict Single Match

```python
from src.models.match_predictor import MatchPredictor

predictor = MatchPredictor.load('models/match_predictor.pkl')
result = predictor.predict_match(
    team_a='Brazil',
    team_b='Argentina',
    stage='quarter_finals',
    venue='USA'
)
print(f"Win probability: {result['win_prob']:.2%}")
```

### Simulate Tournament

```python
from src.models.simulator import TournamentSimulator

simulator = TournamentSimulator(n_simulations=10000)
results = simulator.run()
print(results.get_winner_odds())
```

## 🤝 Contributing

This is a prediction system - improvements welcome in:

- Feature engineering
- Model selection and tuning
- Visualization enhancements
- Data collection automation

## 📄 License

Part of the larger Soccer Prediction project.

## 📧 Contact

For questions about the World Cup prediction module, refer to the main project README.

---

**Last Updated**: January 22, 2026  
**Status**: Planning & Initial Setup  
**Next Step**: Data collection for qualified teams
