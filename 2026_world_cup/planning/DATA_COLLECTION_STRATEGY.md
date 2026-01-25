# Data Collection Strategy for 2026 World Cup Predictions

## Overview

This document details the specific data collection requirements, sources, methods, and timelines for gathering all necessary data for World Cup predictions.

---

## 1. Tournament Structure & Qualification Data

### 1.1 Qualified Teams (48 teams)

**Status**: TO COLLECT

**Sources**:

- **Primary**: [FIFA Official Website](https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026)
- **Backup**: Wikipedia, ESPN, BBC Sport

**Data Points**:

```
- team_name
- fifa_code (e.g., BRA, ARG, USA)
- confederation (UEFA, CONMEBOL, CONCACAF, CAF, AFC, OFC)
- qualification_method (automatic, playoff, host)
- world_cup_appearances (historical count)
- best_finish (historical)
```

**Collection Method**:

- Manual collection from official FIFA announcements
- Verify with multiple sources
- Update as qualifications finalize (June 2025 - March 2026)

**File**: `data/raw/qualified_teams.csv`

### 1.2 Tournament Structure

**Status**: TO COLLECT

**Sources**:

- FIFA official tournament regulations
- Match schedule announcements

**Data Points**:

```json
{
  "groups": {
    "Group A": ["Team1", "Team2", "Team3"],
    "Group B": [...],
    ...
  },
  "venues": [
    {"city": "New York", "stadium": "MetLife Stadium", "capacity": 82500, "altitude": 10},
    ...
  ],
  "schedule": [
    {"match_id": 1, "date": "2026-06-11", "group": "A", "team_a": "...", "team_b": "...", "venue": "..."},
    ...
  ]
}
```

**File**: `data/raw/tournament_structure.json`

---

## 2. National Team Squads

### 2.1 Squad Rosters

**Status**: TO COLLECT (May-June 2026)

**Sources**:

- **Primary**: [Transfermarkt](https://www.transfermarkt.com/)
  - URL pattern: `https://www.transfermarkt.com/[team-name]/startseite/verein/[team-id]`
- **Secondary**: FIFA official team pages
- **Tertiary**: National FA websites

**Data Points**:

```
- player_name
- national_team
- position (GK, DF, MF, FW)
- club_team
- age
- caps (international appearances)
- goals (international)
- club_league
- market_value
```

**Collection Method**:

```python
# Pseudo-code strategy
for team in qualified_teams:
    roster = scrape_transfermarkt(team)
    # Match player names with our database
    matched_players = fuzzy_match(roster, player_predictions)
    save_roster(team, matched_players)
```

**Challenges**:

- Name matching (different spellings, formats)
- Players not in our club database
- Last-minute squad changes

**File**: `data/raw/squad_rosters.csv`

### 2.2 Player Name Matching Strategy

**Approach**:

1. **Exact Match**: Direct name comparison
2. **Fuzzy Match**: Using Levenshtein distance (threshold: 85%)
3. **Manual Review**: For ambiguous cases
4. **External IDs**: Use Transfermarkt IDs, FIFA IDs if available

**Fallback for Unmatched Players**:

- Use league average for their position
- Estimate based on age and club tier
- Flag as "high uncertainty"

---

## 3. Team Performance Data

### 3.1 FIFA Rankings

**Status**: EASY TO COLLECT (Historical data available)

**Sources**:

- **Primary**: [FIFA Rankings Archive](https://www.fifa.com/fifa-world-ranking/)
- **Backup**: Kaggle datasets, ELO ratings

**Data Points**:

```
- ranking_date
- team_name
- fifa_ranking
- fifa_points
- previous_points
- confederation_rank
```

**Collection Method**:

- Scrape monthly rankings from 2006-2026
- Alternative: Download pre-existing Kaggle dataset
- Update monthly until tournament

**File**: `data/raw/fifa_rankings.csv`

**Frequency**: Monthly (updated around 25th of each month)

### 3.2 Recent International Match Results

**Status**: TO COLLECT

**Sources**:

- **Primary**: [FIFA Official Results](https://www.fifa.com/tournaments/)
- **Secondary**: [11v11.com](https://www.11v11.com/)
- **Tertiary**: [SoccerWay](https://www.soccerway.com/)

**Data Points**:

```
- match_date
- team_home
- team_away
- score_home
- score_away
- competition (WC Qualifier, Friendly, Continental Championship)
- venue
- attendance
- neutral_venue (boolean)
```

**Time Range**:

- Core: 2022-2026 (4 years)
- Extended: 2018-2026 (8 years for historical context)

**Collection Method**:

```python
# Strategy
for team in qualified_teams:
    matches = get_team_matches(team, start_date='2022-01-01', end_date='2026-06-01')
    filter_official_matches(matches)  # Remove club games
    save_matches(team, matches)
```

**File**: `data/raw/recent_matches.csv`

**Priority Competitions**:

1. World Cup Qualifiers (highest weight)
2. Continental Championships (EURO 2024, Copa America 2024, etc.)
3. Nations League / Continental competitions
4. High-profile friendlies
5. Low-priority friendlies (lower weight)

---

## 4. Historical World Cup Data

### 4.1 Past Tournament Matches

**Status**: AVAILABLE (Kaggle/GitHub)

**Sources**:

- Kaggle: "International Football Results from 1872 to 2024"
- GitHub: Various World Cup datasets
- Manual: FIFA archives

**Data Points**:

```
- tournament_year
- stage (Group, R16, QF, SF, Final)
- match_date
- team_home
- team_away
- score_home
- score_away
- winner
- penalties (if applicable)
- attendance
- venue
```

**Time Range**: 1930-2022 (focus on 2006-2022 for modeling)

**File**: `data/historical/world_cup_matches.csv`

### 4.2 Historical Team Rosters

**Status**: OPTIONAL (Nice-to-have)

**Purpose**:

- Validate team aggregation methods
- Backtest predictions on previous tournaments

**Sources**:

- Transfermarkt historical squads
- Wikipedia (reliable for past tournaments)

**Time Range**: 2006-2022

---

## 5. Context Data

### 5.1 Venue Information

**Status**: TO COLLECT (Once schedule announced)

**Data Points**:

```
- venue_name
- city
- country
- capacity
- altitude_meters
- climate_zone (tropical, temperate, etc.)
- latitude
- longitude
- surface_type
```

**Source**: Official FIFA documentation, Wikipedia

**File**: `data/raw/venues.csv`

**Use Case**:

- Home advantage calculation
- Altitude effects (e.g., Mexico City)
- Travel distance calculations

### 5.2 Head-to-Head Records

**Status**: DERIVE FROM MATCH DATA

**Method**:

```python
# Calculate from recent_matches.csv
for team_pair in all_matchups:
    h2h = calculate_head_to_head(team_a, team_b, lookback_years=10)
    stats = {
        'wins_a': ...,
        'draws': ...,
        'wins_b': ...,
        'goals_for_a': ...,
        'goals_for_b': ...,
        'last_5_results': [...]
    }
```

**File**: `data/processed/head_to_head.csv`

---

## 6. Optional Enhancement Data

### 6.1 Betting Odds (For Calibration)

**Status**: OPTIONAL

**Purpose**:

- Calibrate probability predictions
- Benchmark our model
- Wisdom of crowds

**Sources**:

- Oddschecker
- Betting exchanges (Betfair, Smarkets)

**Timing**: Collect closer to tournament start

### 6.2 Team Tactical Data

**Status**: OPTIONAL (Advanced)

**Data Points**:

- Preferred formation
- Playing style metrics
- Manager information
- Injury reports

**Sources**:

- WhoScored
- FBRef national team pages
- Sports analytics sites

---

## 7. Data Collection Timeline

```
Timeline:
│
├─ NOW (Jan 2026)
│  ├─ ✅ Collect FIFA rankings (2006-2026)
│  ├─ ✅ Download historical WC data (Kaggle)
│  └─ ✅ Identify qualified teams (partial list)
│
├─ Feb-Mar 2026
│  ├─ 🔲 Complete qualified teams list
│  ├─ 🔲 Collect recent match results (2022-2026)
│  └─ 🔲 Build head-to-head database
│
├─ Apr-May 2026
│  ├─ 🔲 Group draw monitoring
│  ├─ 🔲 Venue information collection
│  └─ 🔲 Tournament structure finalization
│
├─ May-Jun 2026 (CRITICAL PERIOD)
│  ├─ 🔲 Squad announcements (collect immediately)
│  ├─ 🔲 Player-team matching
│  ├─ 🔲 Final roster updates
│  └─ 🔲 Last-minute data updates
│
└─ Jun 2026 (Tournament Start)
   └─ 🔲 Real-time updates (if doing live predictions)
```

---

## 8. Data Quality Checks

### Validation Steps:

1. **Completeness Check**:
   - All 48 teams have rosters ✓
   - All teams have FIFA rankings ✓
   - Match results cover all qualified teams ✓

2. **Consistency Check**:
   - Team names consistent across datasets
   - Date formats standardized
   - Score formats validated

3. **Accuracy Check**:
   - Cross-reference with multiple sources
   - Verify player-team assignments
   - Check for duplicate entries

4. **Freshness Check**:
   - Data updated within last month
   - Squad lists reflect latest changes
   - Rankings are current

### Automated Validation Script:

```python
def validate_data_quality():
    checks = {
        'teams_count': len(qualified_teams) == 48,
        'rosters_complete': all_teams_have_rosters(),
        'player_matches': match_rate > 0.85,
        'rankings_current': rankings_age < 30_days,
        'matches_sufficient': avg_matches_per_team > 10
    }
    return checks
```

---

## 9. Data Storage Format

### File Formats:

- **CSV**: Tabular data (teams, matches, rankings)
- **JSON**: Structured data (tournament structure, configs)
- **Parquet**: Large datasets (if needed)

### Naming Conventions:

```
raw/
  - qualified_teams_YYYYMMDD.csv
  - squad_rosters_YYYYMMDD.csv
  - fifa_rankings_full.csv
  - international_matches_2022_2026.csv

processed/
  - team_strengths_v1.csv
  - player_team_mapping_final.csv
  - head_to_head_stats.csv
```

### Version Control:

- Keep dated versions of raw data
- Document processing steps
- Track data provenance

---

## 10. Tools & Scripts

### Scraping Tools:

```python
# requests + beautifulsoup
import requests
from bs4 import BeautifulSoup

# For dynamic content (if needed)
from selenium import webdriver

# API clients
import fifaapi  # If available
```

### Data Processing:

```python
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz  # Name matching
import json
import yaml
```

### Utilities:

```python
from pathlib import Path
from datetime import datetime, timedelta
import time  # Rate limiting
from tqdm import tqdm  # Progress bars
```

---

## 11. Ethical Considerations

### Web Scraping:

- Respect robots.txt
- Rate limit requests (2-3 seconds between calls)
- Use official APIs when available
- Provide user agent information
- Don't overload servers

### Data Attribution:

- Cite all data sources
- Respect licensing (especially Transfermarkt, FBRef)
- Non-commercial use only (if applicable)

---

## 12. Risk Mitigation

### Potential Issues:

1. **Squad Changes**:
   - Monitor until squad submission deadline
   - Keep backup rosters
2. **Website Structure Changes**:
   - Build flexible scrapers
   - Have manual backup plan
3. **Missing Player Data**:
   - Prepare fallback strategies
   - Use league/position averages
4. **Data Access Issues**:
   - Mirror critical data locally
   - Have multiple source options

---

## Next Actions (Prioritized)

### Week 1:

1. ✅ Document this strategy
2. 🔲 Download FIFA rankings dataset
3. 🔲 Find Kaggle historical WC data
4. 🔲 Set up data folder structure

### Week 2:

1. 🔲 Write Transfermarkt scraper (test on 1-2 teams)
2. 🔲 Build player name matching function
3. 🔲 Create qualified teams list (current status)

### Week 3:

1. 🔲 Collect recent international matches
2. 🔲 Test player-team matching pipeline
3. 🔲 Validate data quality

---

**Last Updated**: January 22, 2026  
**Status**: Planning Phase  
**Owner**: World Cup Prediction Team
