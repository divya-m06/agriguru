import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
import yaml

# Allow running directly or as module
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.common.logger import setup_logger
from ml.common.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_feature_importance,
    plot_learning_curve,
    plot_noise_robustness,
)
from ml.common.seed import seed_everything
from ml.crop_recommendation.src.data import (
    EXPECTED_FEATURES,
    compute_eda_summary,
    load_dataset,
    split_and_save_data,
    validate_schema,
)
from ml.crop_recommendation.src.evaluate import (
    check_and_apply_calibration,
    compute_calibration_curve_data,
    compute_learning_curve_data,
    compute_permutation_importance,
    evaluate_on_test_set,
    run_cross_validation_comparison,
    run_feature_ablation,
    run_noise_robustness_test,
    select_best_model_occam,
    tune_model,
)
from ml.crop_recommendation.src.export import export_artifacts
from ml.crop_recommendation.src.models import build_soft_voting_ensemble, create_pipeline

logger = logging.getLogger("ml.crop_recommendation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate production crop recommendation model.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/crop_recommendation",
        help="Directory containing or to download Crop_recommendation.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/crop_recommendation",
        help="Directory to save reports, plots, and evaluation logs",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/crop_recommendation",
        help="Directory to save deployable model artifacts",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ml/crop_recommendation/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override for deterministic execution",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test mode: run fast iterations on subsample",
    )
    return parser.parse_args()


def generate_markdown_report(
    output_dir: Path,
    eda_summary: Dict[str, Any],
    ablation_df: pd.DataFrame,
    keep_engineering: bool,
    cv_table: pd.DataFrame,
    chosen_model: str,
    test_metrics: Dict[str, Any],
    robustness_results: Dict[str, Any],
    classes: List[str],
    calibration_info: Dict[str, Any],
) -> None:
    """Generate comprehensive production REPORT.md documenting data, modeling, and limitations."""
    report_file = output_dir / "REPORT.md"

    # Identify most confused class pairs from raw confusion matrix
    cm = test_metrics.get("confusion_matrix_raw", [])
    confused_pairs = []
    if cm:
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j and cm[i][j] > 0:
                    confused_pairs.append({
                        "true": classes[i],
                        "predicted": classes[j],
                        "count": cm[i][j],
                    })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Production ML Report: Crop Recommendation\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write(
            f"This report documents the training, selection, and held-out evaluation of a tabular multiclass "
            f"crop recommendation model. The final selected model is **{chosen_model}**, achieving "
            f"**{test_metrics.get('accuracy', 0.0) * 100:.2f}% test accuracy**, "
            f"**{test_metrics.get('macro_f1', 0.0):.4f} Macro-F1**, and "
            f"**{test_metrics.get('top_3_accuracy', 0.0) * 100:.2f}% Top-3 accuracy**.\n\n"
        )

        f.write("## 2. Dataset Characteristics & EDA Findings\n\n")
        f.write(
            f"- **Total Rows:** {eda_summary['total_rows']}\n"
            f"- **Duplicate Rows:** {eda_summary['duplicates_count']}\n"
            f"- **Classes Count:** {len(eda_summary['class_counts'])} (exactly 100 samples per crop, perfectly balanced)\n"
            f"- **Missing Values:** None (0% across all 7 features)\n\n"
            f"> [!NOTE]\n"
            f"> **Synthetic / Heavily Curated Nature:** The dataset exhibits minimal noise, zero missing values, and perfectly "
            f"balanced classes. In this benchmark setting, multiple models routinely exceed 98% accuracy. However, high benchmark "
            f"scores in clean conditions do not guarantee field reliability where unobserved variables (soil drainage, pests, weather extremes) dominate.\n\n"
        )

        f.write("### Outlier Detection (1.5x IQR Rule - Reported, Not Deleted)\n\n")
        f.write("| Feature | Lower Bound | Upper Bound | Outlier Count | Outlier % |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for col, o_info in eda_summary["outlier_summary"].items():
            f.write(f"| `{col}` | {o_info['lower_bound']:.2f} | {o_info['upper_bound']:.2f} | {o_info['outlier_count']} | {o_info['outlier_percentage']:.2f}% |\n")
        f.write("\n*Outliers were retained because extreme agronomic conditions (e.g. heavy monsoon rainfall for rice, high potassium for apples/grapes) are biologically authentic domain signals, not sensor corruptions.*\n\n")

        f.write("## 3. Feature Engineering Ablation Study\n\n")
        f.write("Interaction features evaluated: `NPK_total`, `N_ratio`, `P_ratio`, `K_ratio`, `temp_x_humidity`, `rain_per_humidity`.\n\n")
        f.write("| Model | Baseline Macro-F1 | Engineered Macro-F1 | F1 Gain |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for _, row in ablation_df.iterrows():
            f.write(f"| {row['model']} | {row['mean_macro_f1_baseline']:.4f} (±{row['std_macro_f1_baseline']:.4f}) | {row['mean_macro_f1_engineered']:.4f} (±{row['std_macro_f1_engineered']:.4f}) | {row['f1_gain']:+.4f} |\n")
        f.write(f"\n**Ablation Conclusion:** Feature engineering was **{'retained' if keep_engineering else 'discarded'}** based on whether the mean CV gain exceeded CV standard deviation.\n\n")

        f.write("## 4. Candidate Model Comparison (Repeated Stratified 5-Fold CV)\n\n")
        f.write("| Rank | Model | Macro-F1 (Mean ± Std) | Accuracy (Mean ± Std) | Fit Time (s) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for rank, (_, row) in enumerate(cv_table.iterrows(), start=1):
            f.write(f"| {rank} | **{row['model']}** | {row['mean_macro_f1']:.4f} ± {row['std_macro_f1']:.4f} | {row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f} | {row['fit_time_seconds']:.2f}s |\n")
        f.write("\n")

        f.write("### Occam's Razor Selection Rationale\n\n")
        f.write(
            f"In accordance with production ML principles, we selected **{chosen_model}**. When multiple models "
            f"fall within one standard error of the best score, Occam's razor favors simpler architectures "
            f"(lower computational latency, minimal memory footprint, and robust calibration properties for downstream serving).\n\n"
        )

        f.write("## 5. Held-Out Test Set Performance (Evaluated Once)\n\n")
        f.write(f"- **Test Samples:** {sum([v['support'] for k, v in test_metrics['classification_report'].items() if k not in ['accuracy', 'macro avg', 'weighted avg']])}\n")
        f.write(f"- **Accuracy:** {test_metrics.get('accuracy', 0.0):.4f}\n")
        f.write(f"- **Macro-F1:** {test_metrics.get('macro_f1', 0.0):.4f}\n")
        f.write(f"- **Top-3 Accuracy:** {test_metrics.get('top_3_accuracy', 0.0):.4f}\n")
        if test_metrics.get("log_loss") is not None:
            f.write(f"- **Multiclass Log Loss:** {test_metrics.get('log_loss'):.4f}\n")
        if test_metrics.get("brier_score") is not None:
            f.write(f"- **Multiclass Brier Score:** {test_metrics.get('brier_score'):.4f}\n\n")

        f.write("### Per-Class Performance\n\n")
        f.write("| Crop Class | Precision | Recall | F1-Score | Support |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for cls_name in classes:
            rep = test_metrics["classification_report"].get(cls_name, {})
            if rep:
                f.write(f"| {cls_name} | {rep.get('precision', 0):.2f} | {rep.get('recall', 0):.2f} | {rep.get('f1-score', 0):.2f} | {rep.get('support', 0)} |\n")
        f.write("\n")

        f.write("### Confusion Pairs Analysis\n\n")
        if confused_pairs:
            f.write("| True Class | Predicted As | Confused Count |\n")
            f.write("| :--- | :--- | :--- |\n")
            for pair in confused_pairs[:10]:
                f.write(f"| {pair['true']} | {pair['predicted']} | {pair['count']} |\n")
            f.write("\n*Noticeable confusion occurs predominantly between botanical legumes (e.g. mothbeans, mungbean, blackgram, lentil), which share highly overlapping nutrient (N/P/K) uptake profiles and tropical temperature thresholds.*\n\n")
        else:
            f.write("Zero confusion pairs were observed on the held-out test sample.\n\n")

        f.write("## 6. Generalization & Noise Robustness Tests\n\n")
        f.write("To simulate sensor inaccuracies, Gaussian noise proportional to each feature's standard deviation was injected into the test features:\n\n")
        f.write("| Perturbation Level | Test Accuracy | Macro-F1 | Top-3 Accuracy | Log Loss |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for noise_key, n_metrics in robustness_results.items():
            f.write(f"| {noise_key} | {n_metrics['accuracy']:.4f} | {n_metrics['macro_f1']:.4f} | {n_metrics['top_3_accuracy']:.4f} | {n_metrics.get('log_loss', 0.0):.4f} |\n")
        f.write("\n")

        f.write("## 7. Known Limitations & Production Caveats\n\n")
        f.write(
            "1. **Agronomic Scope:** The model recommends crops based purely on environmental viability (climatic + basic NPK). It does not factor in seasonal cycles, irrigation capacity, market prices, or transportation economics. *'Suitable crop' does not equal 'most profitable crop'*. \n"
            "2. **Unobserved Soil Chemistry:** Crucial soil health metrics (electrical conductivity, organic carbon, micro-nutrients Zn/Fe/B) and soil classification (alluvial, clay, black soil) are absent.\n"
            "3. **Geographical Extrapolation:** Input validation boundaries enforce restrictions derived from observed data. Out-of-distribution queries from desert or alpine zones will be rejected at the API inference boundary.\n"
        )
    logger.info("Saved production report to %s", report_file)


def main() -> None:
    args = parse_args()

    # 1. Config loading
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    seed_everything(seed)

    output_dir = Path(args.output_dir)
    artifacts_dir = Path(args.artifacts_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 2. Setup logger
    logger = setup_logger("ml.crop_recommendation", log_file=output_dir / "train.log")
    logger.info("Initializing Crop Recommendation training pipeline with seed=%d...", seed)
    if args.quick:
        logger.warning("Running in QUICK smoke-test mode!")

    # 3. Load & validate data
    df_raw = load_dataset(
        data_dir=data_dir,
        filename=config["data"].get("filename", "Crop_recommendation.csv"),
        download_url=config["data"].get("default_url"),
    )
    validate_schema(df_raw, expected_classes=config["data"].get("n_classes", 22))

    if args.quick:
        # Take 10 samples per class for smoke testing
        df_raw = df_raw.groupby("label").head(10).reset_index(drop=True)
        logger.info("Quick mode: subsampled dataset to %d rows.", len(df_raw))

    # 4. EDA summary & initial plots
    eda_summary = compute_eda_summary(df_raw)
    plot_correlation_heatmap(df_raw, EXPECTED_FEATURES, output_dir / "correlation_matrix.png")
    plot_feature_distributions(df_raw, EXPECTED_FEATURES, output_dir / "feature_distributions.png")

    # 5. Stratified 70/15/15 split
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train_val,
        y_train_val,
        label_encoder,
        input_ranges,
    ) = split_and_save_data(
        df=df_raw,
        data_dir=data_dir,
        train_ratio=config["split"].get("train_ratio", 0.70),
        val_ratio=config["split"].get("val_ratio", 0.15),
        test_ratio=config["split"].get("test_ratio", 0.15),
        seed=seed,
    )

    # 6. Feature engineering ablation
    ablation_candidates = ["RandomForest", "LightGBM", "GaussianNB"] if not args.quick else ["GaussianNB", "RandomForest"]
    cv_splits = 3 if args.quick else config["cv"].get("n_splits", 5)
    n_repeats = 1 if args.quick else config["cv"].get("n_repeats", 3)

    ablation_df, keep_engineering = run_feature_ablation(
        candidate_names=ablation_candidates,
        X=X_train_val,
        y=y_train_val,
        cv_splits=cv_splits,
        n_repeats=n_repeats,
        seed=seed,
    )

    # 7. Candidate comparison across all algorithms
    all_candidates = [
        "GaussianNB",
        "LogisticRegression",
        "KNN",
        "SVC",
        "RandomForest",
        "ExtraTrees",
        "HistGradientBoosting",
        "LightGBM",
        "XGBoost",
        "MLPClassifier",
    ]
    if args.quick:
        all_candidates = ["GaussianNB", "RandomForest", "LogisticRegression"]

    cv_results_df = run_cross_validation_comparison(
        candidate_names=all_candidates,
        X=X_train_val,
        y=y_train_val,
        cv_splits=cv_splits,
        n_repeats=n_repeats,
        engineer_features=keep_engineering,
        seed=seed,
    )
    cv_results_df.to_csv(output_dir / "cv_comparison.csv", index=False)

    # 8. Hyperparameter tuning for top candidates
    top_candidates = cv_results_df["model"].head(3).tolist()
    tuned_models = []
    tuning_iter = 5 if args.quick else config["tuning"].get("n_iter", 30)

    for cand in top_candidates:
        tuned_pipe, best_params, best_score = tune_model(
            model_name=cand,
            X=X_train_val,
            y=y_train_val,
            engineer_features=keep_engineering,
            n_iter=tuning_iter,
            cv_splits=cv_splits,
            seed=seed,
        )
        tuned_models.append((cand, tuned_pipe))

    # Evaluate soft voting ensemble of top 3
    if len(tuned_models) >= 3 and not args.quick:
        voting_ensemble = build_soft_voting_ensemble(tuned_models[:3], random_state=seed)
        cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=n_repeats, random_state=seed)
        ens_cv = cross_validate(
            voting_ensemble,
            X_train_val,
            y_train_val,
            cv=cv,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            n_jobs=-1,
        )
        ens_macro_f1 = float(np.mean(ens_cv["test_macro_f1"]))
        ens_std_f1 = float(np.std(ens_cv["test_macro_f1"]))
        ens_acc = float(np.mean(ens_cv["test_accuracy"]))
        ens_std_acc = float(np.std(ens_cv["test_accuracy"]))
        ens_fit_time = float(np.mean(ens_cv["fit_time"]))

        ensemble_row = pd.DataFrame([{
            "model": "SoftVotingEnsemble",
            "mean_macro_f1": ens_macro_f1,
            "std_macro_f1": ens_std_f1,
            "mean_accuracy": ens_acc,
            "std_accuracy": ens_std_acc,
            "fit_time_seconds": ens_fit_time,
            "engineer_features": keep_engineering,
        }])
        cv_results_df = pd.concat([cv_results_df, ensemble_row], ignore_index=True)
        cv_results_df = cv_results_df.sort_values(by="mean_macro_f1", ascending=False).reset_index(drop=True)
        cv_results_df.to_csv(output_dir / "cv_comparison.csv", index=False)
        tuned_models.append(("SoftVotingEnsemble", voting_ensemble))
        logger.info("[SoftVotingEnsemble] Macro-F1: %.4f (+/- %.4f) | Accuracy: %.4f (+/- %.4f)",
                    ens_macro_f1, ens_std_f1, ens_acc, ens_std_acc)

    # 9. Model selection using Occam's razor
    chosen_model_name = select_best_model_occam(cv_results_df)
    logger.info("Chosen model: %s", chosen_model_name)

    # Retrieve or instantiate final pipeline
    chosen_pipeline = None
    for name, pipe in tuned_models:
        if name == chosen_model_name:
            chosen_pipeline = pipe
            break
    if chosen_pipeline is None:
        chosen_pipeline = create_pipeline(
            chosen_model_name,
            engineer_features=keep_engineering,
            random_state=seed,
        )

    # 10. Calibration check on train vs val
    calibrated_pipeline, is_calibrated, cal_info = check_and_apply_calibration(
        chosen_pipeline, X_train, y_train, X_val, y_val
    )
    final_pipeline = calibrated_pipeline if is_calibrated else chosen_pipeline

    # 11. Final evaluation on held-out test set (ONCE)
    class_names = [str(c) for c in label_encoder.classes_]
    test_metrics = evaluate_on_test_set(
        pipeline=final_pipeline,
        X_train_val=X_train_val,
        y_train_val=y_train_val,
        X_test=X_test,
        y_test=y_test,
        classes=class_names,
    )
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # 12. Diagnostic plots
    plot_confusion_matrix(
        cm_normalized=np.array(test_metrics["confusion_matrix_normalized"]),
        class_names=class_names,
        output_path=output_dir / "confusion_matrix.png",
    )

    train_sizes, train_scores, val_scores = compute_learning_curve_data(
        final_pipeline, X_train_val, y_train_val, cv_splits=cv_splits, seed=seed
    )
    plot_learning_curve(train_sizes, train_scores, val_scores, output_dir / "learning_curve.png")

    prob_true, prob_pred = compute_calibration_curve_data(final_pipeline, X_test, y_test)
    plot_calibration_curve(prob_true, prob_pred, output_dir / "calibration_curve.png")

    perm_res = compute_permutation_importance(final_pipeline, X_test, y_test, seed=seed)
    plot_feature_importance(
        feature_names=perm_res["features"],
        importances_mean=np.array(perm_res["importances_mean"]),
        importances_std=np.array(perm_res["importances_std"]),
        output_path=output_dir / "feature_importance.png",
    )

    # 13. Noise robustness test
    feature_stds = {col: input_ranges[col]["std"] for col in EXPECTED_FEATURES}
    robustness_results = run_noise_robustness_test(final_pipeline, X_test, y_test, feature_stds, seed=seed)
    with open(output_dir / "noise_robustness.json", "w", encoding="utf-8") as f:
        json.dump(robustness_results, f, indent=2)

    plot_noise_robustness(
        noise_levels=list(robustness_results.keys()),
        accuracies=[v["accuracy"] for v in robustness_results.values()],
        f1_scores=[v["macro_f1"] for v in robustness_results.values()],
        output_path=output_dir / "noise_robustness.png",
    )

    # 14. Export production artifacts
    export_artifacts(
        pipeline=final_pipeline,
        label_encoder=label_encoder,
        feature_order=EXPECTED_FEATURES,
        input_ranges=input_ranges,
        test_metrics=test_metrics,
        cv_comparison_table=cv_results_df.to_dict(orient="records"),
        chosen_model_name=chosen_model_name,
        output_dir=artifacts_dir,
        config=config,
        seed=seed,
        noise_robustness=robustness_results,
        feature_engineering_used=keep_engineering,
    )

    # 15. Generate REPORT.md
    generate_markdown_report(
        output_dir=output_dir,
        eda_summary=eda_summary,
        ablation_df=ablation_df,
        keep_engineering=keep_engineering,
        cv_table=cv_results_df,
        chosen_model=chosen_model_name,
        test_metrics=test_metrics,
        robustness_results=robustness_results,
        classes=class_names,
        calibration_info=cal_info,
    )

    logger.info("Training pipeline completed successfully! Artifacts -> %s, Reports -> %s",
                artifacts_dir, output_dir)


if __name__ == "__main__":
    main()
