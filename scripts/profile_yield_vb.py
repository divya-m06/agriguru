"""
STEP 1: Profile a single ExtraTrees fit+predict on Variant B vs Variant A.
Loads the existing training split, runs ONE fold, prints top-15 slowest calls.
Runs in < 3 minutes total.
"""
import sys, time
from pathlib import Path
import cProfile, pstats, io
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.yield_prediction.src.data import clean_and_validate_data, load_raw_dataset, split_temporal_data
from ml.yield_prediction.src.models import create_yield_pipeline

DATA_DIR = Path("data/yield_prediction")
TARGET = "hg/ha_yield"

print("Loading data...")
raw = load_raw_dataset(DATA_DIR, "yield_df.csv", "")
df, _ = clean_and_validate_data(raw)
df_train, df_val, df_test, df_tv, input_ranges, valid_cats = split_temporal_data(
    df, data_dir=DATA_DIR,
    train_end_year=2008, val_start_year=2009, val_end_year=2010,
    test_start_year=2011, test_end_year=2013,
)
print(f"Train rows: {len(df_train)}, Val rows: {len(df_val)}, Test rows: {len(df_test)}")

X_tr = df_train.drop(columns=[TARGET])
y_tr = df_train[TARGET].to_numpy()
X_va = df_val.drop(columns=[TARGET])
y_va = df_val[TARGET].to_numpy()

ET_PARAMS = dict(n_estimators=150, max_features=0.5, min_samples_leaf=2, random_state=42, n_jobs=-1)

def profile_one(label: str, include_area: bool):
    pipe = create_yield_pipeline(
        "ExtraTrees",
        estimator_params={k: v for k, v in ET_PARAMS.items() if k not in ("random_state", "n_jobs")},
        include_area=include_area,
        random_state=42,
    )
    # Set n_jobs on the regressor
    pipe.set_params(**{"regressor__regressor__n_jobs": -1,
                       "regressor__regressor__n_estimators": 150,
                       "regressor__regressor__max_features": 0.5,
                       "regressor__regressor__min_samples_leaf": 2})

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    pipe.fit(X_tr, y_tr)
    pipe.predict(X_va)
    pr.disable()
    elapsed = time.perf_counter() - t0

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(15)

    print(f"\n{'='*70}")
    print(f"  {label}  —  Total elapsed: {elapsed:.2f}s")
    print(f"{'='*70}")
    print(s.getvalue())
    return elapsed

t_a = profile_one("Variant A (no Area)", include_area=False)
t_b = profile_one("Variant B (with Area)", include_area=True)

print(f"\nSUMMARY")
print(f"  Variant A elapsed : {t_a:.2f}s")
print(f"  Variant B elapsed : {t_b:.2f}s")
print(f"  Speed ratio B/A   : {t_b/t_a:.1f}x")
