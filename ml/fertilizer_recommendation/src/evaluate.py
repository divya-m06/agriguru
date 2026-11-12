import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline

from ml.common.metrics import (
    calculate_classification_metrics,
    calculate_confidence_interval,
)
from ml.fertilizer_recommendation.src.models import (
    create_fertilizer_pipeline,
    get_fertilizer_tuning_grid,
)

logger = logging.getLogger("ml.fertilizer_recommendation.evaluate")


def evaluate_candidates_repeated_cv(
    candidate_names: List[str],
    df: pd.DataFrame,
    target_col: str = "fertilizer_label",
    n_splits: int = 5,
    n_repeats: int = 5,
    add_engineered_features: bool = False,
    seed: int = 42,
    cache_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Evaluate candidate classifiers under Repeated Stratified 5-Fold Cross-Validation with stage caching.

    Args:
        candidate_names: List of model identifiers.
        df: Input DataFrame.
        target_col: Target column name.
        n_splits: Folds per repeat.
        n_repeats: Number of repetitions.
        add_engineered_features: Feature ablation flag.
        seed: Random seed.
        cache_dir: Optional caching directory.

    Returns:
        DataFrame ranking candidates by macro-F1 with 95% confidence intervals.
    """
    X = df.drop(columns=[target_col, "fertilizer_name"], errors="ignore")
    y = df[target_col].to_numpy()

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    results = []

    cache_path = Path(cache_dir) / "candidates" if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)

    logger.info("Evaluating %d candidate models under Repeated Stratified CV (%d folds x %d repeats = %d evaluations)...",
                len(candidate_names), n_splits, n_repeats, n_splits * n_repeats)

    for name in candidate_names:
        item_cache_file = cache_path / f"{name}.json" if cache_path else None
        if item_cache_file and item_cache_file.exists():
            try:
                with open(item_cache_file, "r", encoding="utf-8") as f:
                    scores = json.load(f)
                results.append(scores)
                logger.info("[CACHED] [%s] Macro-F1: %.4f (+/- %.4f) [95%% CI: %.4f - %.4f] | Accuracy: %.4f | Top-3 Acc: %.4f",
                            name, scores["macro_f1_mean"], scores["macro_f1_std"],
                            scores["macro_f1_ci_lower"], scores["macro_f1_ci_upper"],
                            scores["accuracy_mean"], scores["top_3_accuracy_mean"])
                continue
            except Exception as e:
                logger.warning("Failed to load cache for %s: %s. Recomputing...", name, e)

        acc_scores = []
        f1_scores = []
        top3_scores = []

        pipe = create_fertilizer_pipeline(
            model_name=name,
            add_engineered_features=add_engineered_features,
            random_state=seed,
        )

        n_classes = len(np.unique(y))
        labels_list = list(range(n_classes))

        for train_idx, val_idx in rskf.split(X, y):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_va)

            acc_scores.append(float(accuracy_score(y_va, preds)))
            f1_scores.append(float(f1_score(y_va, preds, average="macro", zero_division=0)))

            # Top-3 accuracy
            if hasattr(pipe, "predict_proba"):
                probs = pipe.predict_proba(X_va)
                top3_scores.append(float(top_k_accuracy_score(y_va, probs, k=min(3, n_classes), labels=labels_list)))
            else:
                top3_scores.append(float(accuracy_score(y_va, preds)))

        acc_ci = calculate_confidence_interval(acc_scores)
        f1_ci = calculate_confidence_interval(f1_scores)
        top3_ci = calculate_confidence_interval(top3_scores)

        scores = {
            "model": name,
            "accuracy_mean": acc_ci["mean"],
            "accuracy_std": acc_ci["std"],
            "accuracy_sem": acc_ci["sem"],
            "accuracy_ci_lower": acc_ci["ci_lower"],
            "accuracy_ci_upper": acc_ci["ci_upper"],
            "macro_f1_mean": f1_ci["mean"],
            "macro_f1_std": f1_ci["std"],
            "macro_f1_sem": f1_ci["sem"],
            "macro_f1_ci_lower": f1_ci["ci_lower"],
            "macro_f1_ci_upper": f1_ci["ci_upper"],
            "top_3_accuracy_mean": top3_ci["mean"],
            "top_3_accuracy_std": top3_ci["std"],
        }
        results.append(scores)

        if item_cache_file:
            with open(item_cache_file, "w", encoding="utf-8") as f:
                json.dump(scores, f, indent=2)

        logger.info("[%s] Macro-F1: %.4f (+/- %.4f) [95%% CI: %.4f - %.4f] | Accuracy: %.4f | Top-3 Acc: %.4f",
                    name, scores["macro_f1_mean"], scores["macro_f1_std"],
                    scores["macro_f1_ci_lower"], scores["macro_f1_ci_upper"],
                    scores["accuracy_mean"], scores["top_3_accuracy_mean"])

    df_res = pd.DataFrame(results).sort_values(by="macro_f1_mean", ascending=False).reset_index(drop=True)
    return df_res


def run_nested_cv_tuning(
    candidate_name: str,
    df: pd.DataFrame,
    target_col: str = "fertilizer_label",
    outer_splits: int = 5,
    outer_repeats: int = 3,
    inner_splits: int = 3,
    n_iter: int = 20,
    seed: int = 42,
    cache_dir: Optional[Path | str] = None,
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, float]]:
    """Execute rigorous Nested Cross-Validation (outer repeated 5-fold, inner 3-fold) with caching.

    Returns unbiased generalization score from outer folds and refits on full dataset with best parameters.
    """
    cache_file = Path(cache_dir) / f"nested_tuning_{candidate_name}.json" if cache_dir else None
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            best_params = cached_data["best_params"]
            nested_metrics = cached_data["nested_metrics"]
            logger.info("[CACHED] Loaded nested CV tuning for %s (Outer Macro-F1: %.4f)",
                        candidate_name, nested_metrics["nested_macro_f1_mean"])

            # Refit on full dataset with best params
            X = df.drop(columns=[target_col, "fertilizer_name"], errors="ignore")
            y = df[target_col].to_numpy()
            full_pipe = create_fertilizer_pipeline(candidate_name, random_state=seed)
            full_pipe.set_params(**best_params)
            full_pipe.fit(X, y)
            return full_pipe, best_params, nested_metrics
        except Exception as e:
            logger.warning("Failed to load nested tuning cache for %s: %s. Retuning...", candidate_name, e)

    X = df.drop(columns=[target_col, "fertilizer_name"], errors="ignore")
    y = df[target_col].to_numpy()
    param_dist = get_fertilizer_tuning_grid(candidate_name)

    logger.info("Executing Nested CV on %s (outer: %d folds x %d repeats, inner: %d folds, n_iter=%d)...",
                candidate_name, outer_splits, outer_repeats, inner_splits, n_iter)

    outer_cv = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=seed)
    outer_acc_scores = []
    outer_f1_scores = []
    outer_top3_scores = []
    outer_best_params_list = []

    n_classes = len(np.unique(y))
    labels_list = list(range(n_classes))

    for outer_fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        X_outer_tr, X_outer_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_outer_tr, y_outer_te = y[tr_idx], y[te_idx]

        base_pipe = create_fertilizer_pipeline(candidate_name, random_state=seed)
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed + outer_fold_idx)

        if param_dist:
            search = RandomizedSearchCV(
                base_pipe,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring="f1_macro",
                cv=inner_cv,
                random_state=seed + outer_fold_idx,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_outer_tr, y_outer_tr)
            best_fold_pipe = search.best_estimator_
            outer_best_params_list.append(search.best_params_)
        else:
            base_pipe.fit(X_outer_tr, y_outer_tr)
            best_fold_pipe = base_pipe
            outer_best_params_list.append({})

        preds = best_fold_pipe.predict(X_outer_te)
        outer_acc_scores.append(float(accuracy_score(y_outer_te, preds)))
        outer_f1_scores.append(float(f1_score(y_outer_te, preds, average="macro", zero_division=0)))

        if hasattr(best_fold_pipe, "predict_proba"):
            probs = best_fold_pipe.predict_proba(X_outer_te)
            outer_top3_scores.append(float(top_k_accuracy_score(y_outer_te, probs, k=min(3, n_classes), labels=labels_list)))
        else:
            outer_top3_scores.append(float(accuracy_score(y_outer_te, preds)))

    f1_ci = calculate_confidence_interval(outer_f1_scores)
    acc_ci = calculate_confidence_interval(outer_acc_scores)
    top3_ci = calculate_confidence_interval(outer_top3_scores)

    nested_metrics = {
        "nested_macro_f1_mean": f1_ci["mean"],
        "nested_macro_f1_std": f1_ci["std"],
        "nested_macro_f1_ci_lower": f1_ci["ci_lower"],
        "nested_macro_f1_ci_upper": f1_ci["ci_upper"],
        "nested_accuracy_mean": acc_ci["mean"],
        "nested_accuracy_std": acc_ci["std"],
        "nested_top_3_accuracy_mean": top3_ci["mean"],
    }

    # Select final best parameters by tuning across full dataset
    final_pipe = create_fertilizer_pipeline(candidate_name, random_state=seed)
    if param_dist:
        full_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
        final_search = RandomizedSearchCV(
            final_pipe,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="f1_macro",
            cv=full_cv,
            random_state=seed,
            n_jobs=-1,
            refit=True,
        )
        final_search.fit(X, y)
        final_best_params = final_search.best_params_
        champion_pipeline = final_search.best_estimator_
    else:
        final_pipe.fit(X, y)
        final_best_params = {}
        champion_pipeline = final_pipe

    logger.info("Nested CV Results for %s: Macro-F1 = %.4f (+/- %.4f) [95%% CI: %.4f - %.4f] | Accuracy = %.4f",
                candidate_name, nested_metrics["nested_macro_f1_mean"], nested_metrics["nested_macro_f1_std"],
                nested_metrics["nested_macro_f1_ci_lower"], nested_metrics["nested_macro_f1_ci_upper"],
                nested_metrics["nested_accuracy_mean"])

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"best_params": final_best_params, "nested_metrics": nested_metrics}, f, indent=2)

    return champion_pipeline, final_best_params, nested_metrics


def run_label_permutation_test(
    pipeline: Pipeline,
    df: pd.DataFrame,
    target_col: str = "fertilizer_label",
    n_permutations: int = 200,
    n_splits: int = 5,
    seed: int = 42,
    cache_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Execute label permutation test (200 shuffles) to rigorously prove model beats random chance."""
    cache_file = Path(cache_dir) / "permutation_test.json" if cache_dir else None
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                res = json.load(f)
            logger.info("[CACHED] Loaded permutation test: Observed Macro-F1 = %.4f, p-value = %.4f",
                        res["observed_score"], res["p_value"])
            return res
        except Exception:
            pass

    X = df.drop(columns=[target_col, "fertilizer_name"], errors="ignore")
    y = df[target_col].to_numpy()

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # 1. Compute observed score
    observed_f1s = []
    for tr_idx, te_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_te)
        observed_f1s.append(float(f1_score(y_te, preds, average="macro", zero_division=0)))
    observed_score = float(np.mean(observed_f1s))

    # 2. Run label permutations
    logger.info("Running Label Permutation Test (%d iterations)...", n_permutations)
    rng = np.random.RandomState(seed)
    null_scores = []

    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_f1s = []
        for tr_idx, te_idx in cv.split(X, y_perm):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y_perm[tr_idx], y_perm[te_idx]
            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict(X_te)
            perm_f1s.append(float(f1_score(y_te, preds, average="macro", zero_division=0)))
        null_scores.append(float(np.mean(perm_f1s)))

    # Empirical p-value
    count_better = int(np.sum(np.array(null_scores) >= observed_score))
    p_value = (count_better + 1.0) / (n_permutations + 1.0)

    result = {
        "observed_score": observed_score,
        "null_scores": null_scores,
        "null_mean": float(np.mean(null_scores)),
        "null_std": float(np.std(null_scores)),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
    }

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    logger.info("Permutation test complete: Observed Macro-F1 = %.4f | Null Mean = %.4f | p-value = %.4f (%s)",
                observed_score, result["null_mean"], p_value, "Significant (p < 0.01)" if p_value < 0.01 else "Not Significant")

    return result


def run_out_of_fold_evaluation(
    pipeline: Pipeline,
    df: pd.DataFrame,
    class_names: List[str],
    target_col: str = "fertilizer_label",
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate accumulated out-of-fold probability distribution and aggregate classification metrics."""
    X = df.drop(columns=[target_col, "fertilizer_name"], errors="ignore")
    y = df[target_col].to_numpy()

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    n_samples = len(df)
    n_classes = len(class_names)

    # Accumulate out-of-fold probability predictions across repeats
    prob_accumulator = np.zeros((n_samples, n_classes), dtype=float)
    count_accumulator = np.zeros(n_samples, dtype=int)

    for tr_idx, te_idx in rskf.split(X, y):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, _ = y[tr_idx], y[te_idx]

        pipeline.fit(X_tr, y_tr)
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_te)
        else:
            preds = pipeline.predict(X_te)
            probs = np.zeros((len(preds), n_classes))
            probs[np.arange(len(preds)), preds] = 1.0

        prob_accumulator[te_idx] += probs
        count_accumulator[te_idx] += 1

    oof_probs = prob_accumulator / count_accumulator[:, np.newaxis]
    oof_preds = np.argmax(oof_probs, axis=1)

    metrics = calculate_classification_metrics(
        y_true=y,
        y_pred=oof_preds,
        y_prob=oof_probs,
        classes=class_names,
        k=3,
    )
    metrics["oof_probs"] = oof_probs.tolist()
    metrics["oof_preds"] = oof_preds.tolist()
    return metrics


def select_best_classifier_occam(
    cv_table: pd.DataFrame,
    simplicity_order: Optional[List[str]] = None,
) -> str:
    """Select the simplest model within 1 standard error of the best macro-F1 score."""
    if simplicity_order is None:
        simplicity_order = [
            "LogisticRegression",
            "GaussianNB",
            "KNN",
            "DecisionTree",
            "SVC",
            "RandomForest",
            "ExtraTrees",
            "XGBoost",
            "CatBoost",
        ]

    # Filter out dummy baselines
    cand_df = cv_table[~cv_table["model"].str.startswith("Dummy")].reset_index(drop=True)
    if cand_df.empty:
        return str(cv_table.iloc[0]["model"])

    best_row = cand_df.iloc[0]
    best_score = best_row["macro_f1_mean"]
    best_sem = best_row["macro_f1_sem"]
    threshold = best_score - best_sem

    eligible = cand_df[cand_df["macro_f1_mean"] >= threshold]["model"].tolist()
    logger.info("Best CV Classifier: %s (Macro-F1=%.4f, SEM=%.4f). 1-SE threshold: %.4f",
                best_row["model"], best_score, best_sem, threshold)
    logger.info("Eligible models within 1-SE: %s", eligible)

    for cand in simplicity_order:
        if cand in eligible:
            logger.info("Selected classifier by Occam's razor: %s", cand)
            return cand

    return str(best_row["model"])


def compute_permutation_importance_classification(
    pipeline: Pipeline,
    df: pd.DataFrame,
    target_col: str = "fertilizer_label",
    n_repeats: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute permutation importance on features with macro-F1 scoring."""
    X = df.drop(columns=[target_col, "fertilizer_name"], errors="ignore")
    y = df[target_col].to_numpy()

    res = permutation_importance(
        pipeline,
        X,
        y,
        scoring="f1_macro",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )
    return {
        "importances_mean": res.importances_mean.tolist(),
        "importances_std": res.importances_std.tolist(),
        "features": list(X.columns),
    }
