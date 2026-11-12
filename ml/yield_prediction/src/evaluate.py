import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import time

from ml.common.metrics import calculate_per_crop_metrics, calculate_regression_metrics
from ml.yield_prediction.src.models import (
    create_quantile_regressors,
    create_yield_pipeline,
    get_yield_tuning_distributions,
)

logger = logging.getLogger("ml.yield_prediction.evaluate")


def generate_expanding_window_splits(
    df: pd.DataFrame,
    start_train_year: int = 1990,
    end_train_year: int = 2008,
    min_train_years: int = 12,
    window_step_years: int = 2,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window temporal CV folds on training data.

    Prevents lookahead leakage while evaluating temporal generalization.
    """
    splits = []
    current_train_end = start_train_year + min_train_years - 1

    while current_train_end + window_step_years <= end_train_year:
        val_start = current_train_end + 1
        val_end = min(current_train_end + window_step_years, end_train_year)

        train_indices = df.index[df["Year"] <= current_train_end].to_numpy()
        val_indices = df.index[(df["Year"] >= val_start) & (df["Year"] <= val_end)].to_numpy()

        if len(train_indices) > 0 and len(val_indices) > 0:
            splits.append((train_indices, val_indices))

        current_train_end = val_end

    return splits


def run_temporal_cv(
    pipeline: Pipeline,
    df_train: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    target_col: str = "hg/ha_yield",
) -> Dict[str, float]:
    """Execute expanding-window temporal CV for a pipeline on training years."""
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    smape_scores = []

    for train_idx, val_idx in splits:
        train_fold = df_train.iloc[train_idx]
        val_fold = df_train.iloc[val_idx]

        X_tr = train_fold.drop(columns=[target_col])
        y_tr = train_fold[target_col].to_numpy()

        X_va = val_fold.drop(columns=[target_col])
        y_va = val_fold[target_col].to_numpy()

        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_va)

        m = calculate_regression_metrics(y_va, preds)
        r2_scores.append(m["r2"])
        rmse_scores.append(m["rmse_hg_ha"])
        mae_scores.append(m["mae_hg_ha"])
        smape_scores.append(m["smape_pct"])

    return {
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "smape_mean": float(np.mean(smape_scores)),
        "smape_std": float(np.std(smape_scores)),
    }


def compare_candidates_cv(
    candidate_names: List[str],
    df_train: pd.DataFrame,
    include_area: bool = False,
    include_year: bool = True,
    use_rainfall_x_temp: bool = False,
    use_climate_anomaly: bool = False,
    seed: int = 42,
    cache_dir: Optional[Path | str] = None,
    code_hash: str = "",
) -> pd.DataFrame:
    """Evaluate candidate regressors under expanding-window temporal CV with stage caching."""
    splits = generate_expanding_window_splits(df_train)
    results = []

    cache_path = Path(cache_dir) / "candidates" if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)

    variant_tag = "Variant_B" if include_area else "Variant_A"
    logger.info("Evaluating %d candidate models under Expanding-Window CV (%d temporal folds, %s)...",
                len(candidate_names), len(splits), "Variant B (with Area)" if include_area else "Variant A (without Area)")

    for name in candidate_names:
        cache_filename = f"{variant_tag}_{name}_seed{seed}_{code_hash}.json" if code_hash else f"{variant_tag}_{name}.json"
        item_cache_file = cache_path / cache_filename if cache_path else None
        if item_cache_file and item_cache_file.exists():
            try:
                with open(item_cache_file, "r", encoding="utf-8") as f:
                    scores = json.load(f)
                results.append(scores)
                logger.info("[CACHED] [%s] Temporal CV R2: %.4f (+/- %.4f) | MAE: %.1f hg/ha | sMAPE: %.2f%%",
                            name, scores["r2_mean"], scores["r2_std"], scores["mae_mean"], scores["smape_mean"])
                continue
            except Exception as e:
                logger.warning("Failed to load cache for %s (%s): %s. Recomputing...", name, variant_tag, e)

        t0 = time.perf_counter()
        pipe = create_yield_pipeline(
            model_name=name,
            include_area=include_area,
            include_year=include_year,
            use_rainfall_x_temp=use_rainfall_x_temp,
            use_climate_anomaly=use_climate_anomaly,
            random_state=seed,
        )
        scores = run_temporal_cv(pipe, df_train, splits)
        elapsed = time.perf_counter() - t0
        scores["model"] = name
        scores["include_area"] = include_area
        scores["include_year"] = include_year
        scores["elapsed_seconds"] = round(elapsed, 2)
        results.append(scores)

        if item_cache_file:
            with open(item_cache_file, "w", encoding="utf-8") as f:
                json.dump(scores, f, indent=2)

        logger.info("[%s] Temporal CV R2: %.4f (+/- %.4f) | MAE: %.1f hg/ha | sMAPE: %.2f%% (Elapsed: %.2fs)",
                    name, scores["r2_mean"], scores["r2_std"], scores["mae_mean"], scores["smape_mean"], elapsed)

    df_res = pd.DataFrame(results).sort_values(by="r2_mean", ascending=False).reset_index(drop=True)
    return df_res


def run_group_kfold_cv(
    pipeline: Pipeline,
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "Area",
    target_col: str = "hg/ha_yield",
    seed: int = 42,
    cache_dir: Optional[Path | str] = None,
    model_name: str = "model",
    code_hash: str = "",
) -> Dict[str, float]:
    """Evaluate generalization to unseen countries using GroupKFold by Area with caching."""
    cache_filename = f"group_kfold_{model_name}_seed{seed}_{code_hash}.json" if code_hash else f"group_kfold_{model_name}.json"
    cache_file = Path(cache_dir) / cache_filename if cache_dir else None
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                res = json.load(f)
            logger.info("[CACHED] Loaded GroupKFold CV results for %s", model_name)
            return res
        except Exception:
            pass

    t0 = time.perf_counter()
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col]

    r2_scores = []
    rmse_scores = []
    mae_scores = []
    smape_scores = []

    for train_idx, test_idx in gkf.split(df, groups=groups):
        train_fold = df.iloc[train_idx]
        test_fold = df.iloc[test_idx]

        X_tr = train_fold.drop(columns=[target_col])
        y_tr = train_fold[target_col].to_numpy()

        X_te = test_fold.drop(columns=[target_col])
        y_te = test_fold[target_col].to_numpy()

        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_te)

        m = calculate_regression_metrics(y_te, preds)
        r2_scores.append(m["r2"])
        rmse_scores.append(m["rmse_hg_ha"])
        mae_scores.append(m["mae_hg_ha"])
        smape_scores.append(m["smape_pct"])

    elapsed = time.perf_counter() - t0
    metrics = {
        "group_kfold_r2_mean": float(np.mean(r2_scores)),
        "group_kfold_r2_std": float(np.std(r2_scores)),
        "group_kfold_mae_mean": float(np.mean(mae_scores)),
        "group_kfold_smape_mean": float(np.mean(smape_scores)),
        "elapsed_seconds": round(elapsed, 2),
    }

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    logger.info("GroupKFold completed in %.2fs", elapsed)
    return metrics


def run_leaky_random_split_cv(
    pipeline: Pipeline,
    df: pd.DataFrame,
    n_splits: int = 3,
    subsample_frac: float = 0.3,
    target_col: str = "hg/ha_yield",
    seed: int = 42,
    cache_dir: Optional[Path | str] = None,
    model_name: str = "model",
    code_hash: str = "",
) -> Dict[str, float]:
    """Explicitly measure random-split CV score to demonstrate leakage magnitude vs temporal split with caching.
    Uses 3 folds on a 30% random subsample of the training rows for high-efficiency estimation."""
    cache_filename = f"leaky_random_{model_name}_seed{seed}_{code_hash}.json" if code_hash else f"leaky_random_{model_name}.json"
    cache_file = Path(cache_dir) / cache_filename if cache_dir else None
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                res = json.load(f)
            logger.info("[CACHED] Loaded Leaky Random CV results for %s", model_name)
            return res
        except Exception:
            pass

    t0 = time.perf_counter()
    eval_df = df.sample(frac=subsample_frac, random_state=seed).reset_index(drop=True) if subsample_frac < 1.0 else df
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    r2_scores = []
    mae_scores = []

    for train_idx, test_idx in kf.split(eval_df):
        train_fold = eval_df.iloc[train_idx]
        test_fold = eval_df.iloc[test_idx]

        X_tr = train_fold.drop(columns=[target_col])
        y_tr = train_fold[target_col].to_numpy()

        X_te = test_fold.drop(columns=[target_col])
        y_te = test_fold[target_col].to_numpy()

        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_te)

        m = calculate_regression_metrics(y_te, preds)
        r2_scores.append(m["r2"])
        mae_scores.append(m["mae_hg_ha"])

    elapsed = time.perf_counter() - t0
    metrics = {
        "random_split_leaky_r2": float(np.mean(r2_scores)),
        "random_split_leaky_mae": float(np.mean(mae_scores)),
        "elapsed_seconds": round(elapsed, 2),
    }

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    logger.info("Leaky Random CV completed in %.2fs", elapsed)
    return metrics


def tune_yield_model(
    model_name: str,
    df_train: pd.DataFrame,
    include_area: bool = False,
    include_year: bool = True,
    use_rainfall_x_temp: bool = False,
    use_climate_anomaly: bool = False,
    n_iter: int = 20,
    seed: int = 42,
    target_col: str = "hg/ha_yield",
    cache_dir: Optional[Path | str] = None,
    code_hash: str = "",
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """Tune hyperparameters using expanding-window temporal CV with caching."""
    variant_tag = "Variant_B" if include_area else "Variant_A"
    cache_filename = f"tuning_{variant_tag}_{model_name}_seed{seed}_{code_hash}.json" if code_hash else f"tuning_{variant_tag}_{model_name}.json"
    cache_file = Path(cache_dir) / cache_filename if cache_dir else None

    base_pipe = create_yield_pipeline(
        model_name,
        include_area=include_area,
        include_year=include_year,
        use_rainfall_x_temp=use_rainfall_x_temp,
        use_climate_anomaly=use_climate_anomaly,
        random_state=seed,
    )
    param_dist = get_yield_tuning_distributions(model_name)

    X = df_train.drop(columns=[target_col])
    y = df_train[target_col].to_numpy()

    if not param_dist:
        base_pipe.fit(X, y)
        return base_pipe, {}, 0.0

    if cache_file and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            best_params = cached_data["best_params"]
            best_score = float(cached_data["best_score"])
            base_pipe.set_params(**best_params)
            base_pipe.fit(X, y)
            logger.info("[CACHED] [%s (%s)] Loaded best params: %s (CV R2: %.4f)",
                        model_name, variant_tag, best_params, best_score)
            return base_pipe, best_params, best_score
        except Exception as e:
            logger.warning("Failed to load tuning cache for %s (%s): %s. Retuning...", model_name, variant_tag, e)

    logger.info("Tuning %s (%s, n_iter=%d)...", model_name, variant_tag, n_iter)
    t0 = time.perf_counter()
    splits = generate_expanding_window_splits(df_train)
    # Estimate on CV with serial folds and tree-level parallelism (or n_jobs=1 on CV)
    search = RandomizedSearchCV(
        base_pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="r2",
        cv=splits,
        random_state=seed,
        n_jobs=1,
        refit=True,
    )

    search.fit(X, y)
    elapsed = time.perf_counter() - t0

    best_params = search.best_params_
    best_score = float(search.best_score_)

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"best_params": best_params, "best_score": best_score, "elapsed_seconds": round(elapsed, 2)}, f, indent=2)

    logger.info("Tuning %s (%s) finished in %.2fs with best CV R2: %.4f", model_name, variant_tag, elapsed, best_score)

    logger.info("[%s (%s)] Best Temporal CV R2: %.4f with params: %s",
                model_name, variant_tag, best_score, best_params)
    return search.best_estimator_, best_params, best_score


def select_best_regressor_occam(
    cv_table: pd.DataFrame,
    simplicity_order: Optional[List[str]] = None,
) -> str:
    """Select the simplest regressor within 1 standard error of the highest CV R2 score."""
    if simplicity_order is None:
        simplicity_order = [
            "Baseline_Mean",
            "Baseline_Median",
            "Ridge",
            "ElasticNet",
            "KNN",
            "RandomForest",
            "ExtraTrees",
            "GradientBoosting",
            "HistGradientBoosting",
            "LightGBM",
            "XGBoost",
            "CatBoost",
            "MLPRegressor",
        ]

    best_row = cv_table.iloc[0]
    best_r2 = best_row["r2_mean"]
    best_std = best_row["r2_std"]
    # Approx 3 temporal splits
    one_se = best_std / np.sqrt(3)
    threshold = best_r2 - one_se

    logger.info("Best CV Regressor: %s (R2=%.4f). 1-SE threshold: %.4f",
                best_row["model"], best_r2, threshold)

    eligible_models = cv_table[cv_table["r2_mean"] >= threshold]["model"].tolist()
    logger.info("Eligible models within 1-SE: %s", eligible_models)

    for cand in simplicity_order:
        if cand in eligible_models:
            logger.info("Selected regressor by Occam's razor: %s", cand)
            return cand

    return str(best_row["model"])


def evaluate_final_test(
    pipeline: Pipeline,
    df_train_val: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = "hg/ha_yield",
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], np.ndarray]:
    """Fit chosen model on train+val (1990-2010) and evaluate ONCE on test set (2011-2013)."""
    X_tr = df_train_val.drop(columns=[target_col])
    y_tr = df_train_val[target_col].to_numpy()

    X_te = df_test.drop(columns=[target_col])
    y_te = df_test[target_col].to_numpy()

    logger.info("Refitting chosen model on train+val (%d samples)...", len(df_train_val))
    pipeline.fit(X_tr, y_tr)

    logger.info("Evaluating on held-out test years 2011-2013 (%d samples)...", len(df_test))
    y_pred = pipeline.predict(X_te)

    # Core metrics
    metrics = calculate_regression_metrics(y_te, y_pred)
    # Per-crop breakdown
    per_crop = calculate_per_crop_metrics(y_te, y_pred, df_test["Item"])

    logger.info("Test Set Results -> R2: %.4f | RMSE: %.1f hg/ha (%.2f t/ha) | MAE: %.1f hg/ha (%.2f t/ha) | sMAPE: %.2f%%",
                metrics["r2"], metrics["rmse_hg_ha"], metrics["rmse_t_ha"],
                metrics["mae_hg_ha"], metrics["mae_t_ha"], metrics["smape_pct"])

    return metrics, per_crop, y_pred


def compute_permutation_importance_regression(
    pipeline: Pipeline,
    df_test: pd.DataFrame,
    target_col: str = "hg/ha_yield",
    seed: int = 42,
    n_repeats: int = 10,
) -> Dict[str, Any]:
    """Compute permutation importance on held-out test split."""
    X_te = df_test.drop(columns=[target_col])
    y_te = df_test[target_col].to_numpy()

    res = permutation_importance(
        pipeline,
        X_te,
        y_te,
        scoring="r2",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )
    return {
        "importances_mean": res.importances_mean.tolist(),
        "importances_std": res.importances_std.tolist(),
        "features": list(X_te.columns),
    }
