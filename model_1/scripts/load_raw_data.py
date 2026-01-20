"""
Load raw data script for soccer prediction project.
"""

import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import pandas as pd
from model_1.config import RAW_TOTAL_DIR, PROCESSED_DIR
def load_raw_data():
    """Load raw data from API."""
    
