# Model Design Decisions & Trade-offs

## Overview

This document outlines key modeling decisions, alternative approaches, and the rationale behind our choices for the 2026 World Cup prediction system.

---

## Decision 1: Player Prediction Model Selection

### Options:

| Option       | Pros                                   | Cons                           | Decision           |
| ------------ | -------------------------------------- | ------------------------------ | ------------------ |
| **Model 1**  | Already trained, familiar structure    | May have lower accuracy        | ✅ **Start Here**  |
| **Model 2**  | More comprehensive features (224 cols) | More complex, needs validation | 🔄 Test Both       |
| **Ensemble** | Best of both worlds                    | More complexity, slower        | 🔄 If time permits |

### Recommendation:

- **Phase 1**: Use Model 1 predictions (simpler, already available)
- **Phase 2**: Compare with Model 2 predictions
- **Phase 3**: Consider ensemble if significant differences

### Implementation:

```python
# config flag
USE_PLAYER_MODEL = "model_1"  # or "model_2" or "ensemble"
```

---

## Decision 2: Team Aggregation Method

### Challenge:

How to convert 23 player predictions into a single team strength score?

### Options Considered:

#### Option A: Simple Average

```python
team_strength = mean([p.rating for p in squad])
```

**Pros**: Simple, interpretable  
**Cons**: Ignores positions, bench quality  
**Use Case**: Baseline only

#### Option B: Weighted Average by Position

```python
weights = {'GK': 1.0, 'DF': 1.2, 'MF': 1.1, 'FW': 1.0}
team_strength = weighted_mean(squad_ratings, position_weights)
```

**Pros**: Accounts for position importance  
**Cons**: Subjective weights  
**Use Case**: Better baseline

#### Option C: Starting XI + Bench Split (⭐ RECOMMENDED)

```python
starting_xi_strength = mean(top_11_players)
bench_strength = mean(next_12_players)
team_strength = 0.7 * starting_xi + 0.3 * bench
```

**Pros**: Captures squad depth, realistic  
**Cons**: Need to select "best XI"  
**Use Case**: Primary approach

#### Option D: Position-Specific Vectors

```python
team_vector = {
    'attack': mean(top_3_forwards),
    'midfield': mean(top_3_midfielders),
    'defense': mean(top_4_defenders + best_gk)
}
```

**Pros**: Captures tactical balance, enables matchup analysis  
**Cons**: More features, complex  
**Use Case**: **BEST APPROACH** - enables attack vs defense interactions

### Decision: **Option D + Option C**

- Use position-specific vectors for detailed features
- Include overall team strength for simplicity
- Test both in feature importance analysis

---

## Decision 3: Starting XI Selection Algorithm

### Challenge:

How to select the best 11 players when positions overlap?

### Approach:

```python
def select_starting_xi(squad, formation='4-3-3'):
    """
    Select best XI using:
    1. Overall rating
    2. Position fit
    3. Form (recent performance)
    4. Age/experience balance
    """
    positions = parse_formation(formation)
    starting_xi = []

    for pos in positions:
        eligible = filter_by_position(squad, pos)
        best = max(eligible, key=lambda p: p.rating)
        starting_xi.append(best)

    return starting_xi
```

### Considerations:

- **Formation**: Assume 4-3-3 or 4-4-2 (most common)
- **Versatility**: Some players can play multiple positions
- **Manager Preference**: Don't have this data (limitation)

### Fallback:

If uncertain, use top 11 by rating with position constraints (min 3 DEF, min 2 MF, etc.)

---

## Decision 4: Match Prediction Model Architecture

### Challenge:

What type of model for predicting match outcomes?

### Options Analysis:

#### 1. Gradient Boosting (XGBoost/LightGBM) ⭐ RECOMMENDED

**Pros**:

- Excellent for tabular data
- Handles interactions automatically
- Feature importance built-in
- Fast training and prediction
- Non-linear relationships

**Cons**:

- Can overfit with small data
- Less interpretable than linear models

**Best For**: Win/Draw/Loss classification

**Implementation**:

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    objective='multi:softprob',  # 3 classes
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05
)
```

#### 2. Poisson Regression

**Pros**:

- Natural for goal counts
- Highly interpretable
- Established in soccer analytics

**Cons**:

- Assumes independence (not always true)
- Linear relationships only

**Best For**: Exact score prediction

**Implementation**:

```python
from statsmodels.api import GLM, families

goals_home = GLM(y_home, X, family=families.Poisson()).fit()
goals_away = GLM(y_away, X, family=families.Poisson()).fit()
```

#### 3. Random Forest

**Pros**:

- Robust, less overfitting
- Handles missing data well

**Cons**:

- Generally less accurate than XGBoost
- Larger model size

**Best For**: Ensemble member

#### 4. Neural Network

**Pros**:

- Can learn complex patterns
- Flexible architecture

**Cons**:

- Needs more data
- Harder to train
- Less interpretable
- Overkill for this problem

**Best For**: Optional experiment

### Decision: **Hybrid Approach**

1. **Primary**: XGBoost for match outcomes (W/D/L)
2. **Secondary**: Poisson regression for goal counts
3. **Ensemble**: Combine if they disagree significantly

**Why Hybrid?**

- XGBoost gives us probabilities: P(win), P(draw), P(loss)
- Poisson gives us expected goals and score distributions
- Can validate one against the other
- More robust predictions

---

## Decision 5: Target Variable Formulation

### For Classification (XGBoost):

#### Option A: Three-Class (W/D/L)

```python
target = {
    'win': 1,
    'draw': 0,
    'loss': -1
}
```

**Use**: Multinomial classification

#### Option B: Binary (Team A Perspective)

```python
target = {
    'team_a_win': 1,
    'team_a_not_win': 0  # (draw or loss)
}
```

**Use**: Simpler, but loses draw information

### Decision: **Three-Class**

- More informative
- Draws are important in tournaments (group stage)
- Can still extract P(win) for Team A

---

## Decision 6: Feature Engineering Strategy

### Tier 1: Essential Features (Must Have)

```python
essential_features = [
    # Team strength
    'team_a_overall_rating',
    'team_b_overall_rating',
    'rating_difference',

    # Rankings
    'team_a_fifa_rank',
    'team_b_fifa_rank',
    'rank_difference',

    # Context
    'home_advantage',  # 0, 0.5, 1
    'stage_importance',  # 0-5

    # Recent form (if available)
    'team_a_recent_win_rate',
    'team_b_recent_win_rate'
]
```

### Tier 2: Enhanced Features (Should Have)

```python
enhanced_features = [
    # Position-specific
    'team_a_attack_rating',
    'team_b_defense_rating',
    'attack_defense_matchup',  # A_attack vs B_defense

    # Squad depth
    'team_a_bench_quality',
    'team_b_bench_quality',

    # Experience
    'team_a_avg_age',
    'team_b_avg_caps',

    # Historical
    'head_to_head_win_rate',
    'confederation_matchup'
]
```

### Tier 3: Advanced Features (Nice to Have)

```python
advanced_features = [
    # Tactical
    'style_compatibility',  # e.g., counter-attack vs possession

    # Momentum
    'recent_goal_difference',
    'streak_indicator',

    # Tournament-specific
    'days_since_last_match',
    'group_stage_position',
    'knockout_pressure'
]
```

### Decision: **Implement Tier 1 & 2, Experiment with Tier 3**

---

## Decision 7: Training Data Strategy

### Challenge:

How much historical data to use?

### Options:

| Time Range                     | # of Matches | Pros          | Cons                     |
| ------------------------------ | ------------ | ------------- | ------------------------ |
| WC only (1930-2022)            | ~900         | Most relevant | Too few samples          |
| WC + Continental (2006-2022)   | ~3000        | Balanced      | Mixed competition levels |
| All internationals (2010-2026) | ~15000       | Lots of data  | Includes friendlies      |

### Decision: **Stratified Approach**

```python
training_data = {
    'world_cup_matches': {
        'weight': 1.0,
        'years': [2006, 2010, 2014, 2018, 2022]
    },
    'continental_championships': {
        'weight': 0.8,  # EURO, Copa America, AFCON, etc.
        'years': [2012, 2016, 2020, 2024]
    },
    'qualifiers': {
        'weight': 0.6,
        'years': [2018, 2022, 2026]
    },
    'friendlies': {
        'weight': 0.3,
        'years': [2022, 2023, 2024, 2025]
    }
}
```

**Rationale**:

- Weight by competition importance
- Focus on recent data (more relevant)
- Include variety for generalization

---

## Decision 8: Handling Imbalanced Data

### Problem:

- More home wins than away wins
- More wins than draws
- Top teams overrepresented

### Solutions:

#### 1. Class Weights

```python
class_weights = {
    'win': 1.0,
    'draw': 1.5,  # Draws are rarer, upweight
    'loss': 1.0
}
```

#### 2. SMOTE (Synthetic Minority Over-sampling)

```python
from imblearn.over_sampling import SMOTE
X_balanced, y_balanced = SMOTE().fit_resample(X, y)
```

#### 3. Stratified Sampling

```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True)
```

### Decision: **Stratified CV + Class Weights**

- SMOTE is overkill for this problem
- Stratification ensures balanced folds
- Class weights handle imbalance naturally

---

## Decision 9: Tournament Simulation Method

### Options:

#### Option A: Deterministic (Most Likely Path)

- Select winner with highest probability for each match
- **Problem**: No uncertainty quantification

#### Option B: Monte Carlo Simulation ⭐ RECOMMENDED

- Run tournament 10,000+ times
- Sample outcomes based on probabilities
- Aggregate results

```python
def simulate_tournament(n_simulations=10000):
    results = []
    for i in range(n_simulations):
        tournament = Tournament()

        # Group stage
        for match in group_matches:
            outcome = sample_outcome(match)
            tournament.record(match, outcome)

        # Knockout stage
        for round in knockout_rounds:
            for match in round.matches:
                outcome = sample_outcome(match)
                if outcome == 'draw':
                    outcome = simulate_penalties()
                tournament.record(match, outcome)

        results.append(tournament.winner)

    return aggregate_results(results)
```

### Decision: **Monte Carlo with 10,000 simulations**

- Provides confidence intervals
- Handles uncertainty properly
- Computationally feasible

---

## Decision 10: Penalty Shootout Modeling

### Challenge:

How to predict penalty outcomes in knockout stages?

### Options:

#### Option A: 50-50 Random

```python
winner = random.choice([team_a, team_b])
```

#### Option B: Slight Favorite Advantage

```python
# Team with better rating gets 55% chance
p_win = 0.5 + 0.05 * (rating_diff / max_rating_diff)
```

#### Option C: Penalty-Specific Model

```python
# Use features like:
- GK penalty save rate
- Team penalty conversion rate
- Pressure experience
```

### Decision: **Option B (Pragmatic)**

- Option A is too simplistic
- Option C requires data we don't have
- Slight advantage based on team strength is reasonable
- Penalties are inherently random anyway

---

## Decision 11: Model Validation Approach

### Strategy:

```python
validation_plan = {
    'historical_validation': {
        'train': [2006, 2010, 2014, 2018],
        'test': [2022],
        'metric': 'accuracy, log_loss, brier_score'
    },
    'cross_validation': {
        'method': 'leave_one_tournament_out',
        'tournaments': [2006, 2010, 2014, 2018, 2022],
        'metric': 'average_log_loss'
    },
    'baseline_comparison': {
        'baselines': ['fifa_rankings', 'elo_ratings', 'betting_odds'],
        'metric': 'relative_improvement'
    }
}
```

### Success Criteria:

- ✅ **Good**: >50% match accuracy (better than random)
- ✅ **Great**: >55% match accuracy (better than betting odds)
- ✅ **Excellent**: Correctly predict 2+ semifinalists

---

## Decision 12: Uncertainty Quantification

### Approaches:

1. **Probability Calibration**:

```python
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic')
```

2. **Confidence Intervals from Simulation**:

```python
winner_counts = count_wins_from_simulations(results)
confidence_interval = np.percentile(winner_counts, [2.5, 97.5])
```

3. **Prediction Intervals**:

```python
# For expected goals
mu, sigma = predict_goals(match)
prediction_interval = (mu - 1.96*sigma, mu + 1.96*sigma)
```

### Decision: **Use All Three**

- Calibration for match probabilities
- Simulation for tournament outcomes
- Prediction intervals for goal counts

---

## Key Trade-offs Summary

| Aspect           | Simple Approach     | Complex Approach  | Our Choice          | Why                         |
| ---------------- | ------------------- | ----------------- | ------------------- | --------------------------- |
| Player Model     | Model 1             | Ensemble          | Model 1 → Test both | Start simple, iterate       |
| Team Aggregation | Mean rating         | Position vectors  | Position vectors    | Better features             |
| Match Model      | Logistic regression | XGBoost + Poisson | XGBoost + Poisson   | Balance complexity/accuracy |
| Training Data    | WC only             | All matches       | Weighted mix        | More data, less noise       |
| Simulation       | Deterministic       | Monte Carlo       | Monte Carlo         | Proper uncertainty          |

---

## Open Questions & Future Experiments

### To Test:

1. Does including friendly matches help or hurt?
2. What's the optimal feature set? (feature selection)
3. Should we use ELO ratings alongside FIFA rankings?
4. Can we incorporate manager experience?
5. Does recent form matter more than overall squad quality?

### Ablation Studies:

```python
experiments = [
    'baseline_fifa_only',
    'baseline_player_ratings_only',
    'full_model_without_h2h',
    'full_model_without_form',
    'full_model_all_features'
]
```

---

## Implementation Priority

### Phase 1 (MVP):

- ✅ Basic team aggregation (Option C)
- ✅ XGBoost match predictor
- ✅ Monte Carlo simulator
- ✅ Historical validation

### Phase 2 (Enhanced):

- 🔄 Position-specific features (Option D)
- 🔄 Poisson goal predictor
- 🔄 Penalty shootout model
- 🔄 Feature engineering expansion

### Phase 3 (Advanced):

- ⏳ Model ensemble
- ⏳ Tactical features
- ⏳ Real-time updates
- ⏳ Interactive dashboard

---

## References & Inspiration

- **Academic**: Dixon & Coles (1997) - Poisson model for soccer scores
- **Industry**: FiveThirtyEight SPI ratings
- **Community**: Kaggle soccer prediction competitions
- **Books**: "The Numbers Game" by Anderson & Sally

---

**Last Updated**: January 22, 2026  
**Status**: Design Phase  
**Next Review**: After MVP implementation
