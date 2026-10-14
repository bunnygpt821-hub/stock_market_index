#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'runtime/nikkei225_yahoo_clean_2016_present.csv'
IDX = ROOT / 'data/indexes/nikkei225/daily.csv'
BASE_PATH = Path(__file__).with_name('paradise_lost_v2_analysis.py')
OUT = ROOT / 'analysis_outputs/paradise_lost_first_squeeze'
OUT.mkdir(parents=True, exist_ok=True)

