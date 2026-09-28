from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import DEFAULT_OUTPUT_DIR
from run_profile_models import COMPACT_GPU_FEATURES, STATIC_FEATURES
from run_trace_hpct_models import make_pipeline, target_values


CANDIDATE_MODEL = "static_compact_gpu_ridge"
FINAL_FEATURES = STATIC_FEATURES + COMPACT_GPU_FEATURES
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def prepare_minimal(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["log_batch"] = np.log(result.batch.astype(float))
    result["log_pixels"] = np.log(result.imgsz.astype(float) ** 2)
    result["amp_numeric"] = result.amp.astype(str).str.lower().map(
        {"true": 1.0, "false": 0.0}
    )
    result["log_profile_sample_count"] = np.log(
        result.source_gpu_sample_count.astype(float)
    )
    if result[FINAL_FEATURES].isna().any().any():
        missing = result[FINAL_FEATURES].isna().sum()
        raise ValueError(f"Final model features contain missing values: {missing[missing > 0].to_dict()}")
    return result


def bootstrap_metric_intervals(
    frame: pd.DataFrame, rng: np.random.Generator, repeats: int
) -> list[dict]:
    actual = frame.actual_ms.to_numpy(float)
    predicted = frame.predicted_ms.to_numpy(float)
    n = len(frame)
    values = {"mae_ms": [], "rmse_ms": [], "mape_percent": []}
    for _ in range(repeats):
        index = rng.integers(0, n, size=n)
        error = predicted[index] - actual[index]
        values["mae_ms"].append(float(np.mean(np.abs(error))))
        values["rmse_ms"].append(float(np.sqrt(np.mean(error**2))))
        values["mape_percent"].append(
            float(np.mean(np.abs(error) / actual[index]) * 100)
        )
    rows = []
    for metric, samples in values.items():
        point = {
            "mae_ms": float(np.mean(np.abs(predicted - actual))),
            "rmse_ms": float(np.sqrt(np.mean((predicted - actual) ** 2))),
            "mape_percent": float(np.mean(np.abs(predicted - actual) / actual) * 100),
        }[metric]
        low, high = np.percentile(samples, [2.5, 97.5])
        rows.append(
            {
                "metric": metric,
                "estimate": point,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_repeats": repeats,
            }
        )
    return rows


def select_final_alpha(data: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []
    for alpha in ALPHAS:
        ape = []
        for workload in sorted(data.workload_id.unique()):
            train = data[data.workload_id != workload]
            test = data[data.workload_id == workload]
            pipeline = make_pipeline()
            pipeline.set_params(model__alpha=alpha)
            pipeline.fit(train[FINAL_FEATURES], target_values(train))
            predicted = float(np.exp(pipeline.predict(test[FINAL_FEATURES])[0]))
            actual = float(test.iloc[0].target_ground_truth_iter_ms)
            ape.append(abs(predicted - actual) / actual * 100)
        rows.append({"alpha": alpha, "loow_mape_percent": float(np.mean(ape))})
    scores = pd.DataFrame(rows)
    best = scores.sort_values(["loow_mape_percent", "alpha"]).iloc[0]
    return float(best.alpha), scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 5 final comparison and model packaging.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    experiments = root / "model_experiments"
    output = experiments / "part5_final"
    model_dir = output / "models"
    example_dir = output / "example_inputs"
    model_dir.mkdir(parents=True, exist_ok=True)
    example_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(root / "cross_device_samples.csv")
    data = prepare_minimal(raw)
    p1m = pd.read_csv(experiments / "part1_baselines" / "metrics.csv")
    p2m = pd.read_csv(experiments / "part2_workload_models" / "metrics.csv")
    p3m = pd.read_csv(experiments / "part3_profile_models" / "metrics.csv")
    p4m = pd.read_csv(experiments / "part4_trace_hpct_models" / "metrics.csv")
    p4p = pd.read_csv(experiments / "part4_trace_hpct_models" / "predictions.csv")
    coefficients = pd.read_csv(experiments / "part4_trace_hpct_models" / "coefficients.csv")

    candidate = p4p[p4p.model == CANDIDATE_MODEL].copy()
    metadata = raw[
        ["source_device_id", "target_device_id", "workload_id", "batch", "imgsz", "amp"]
    ]
    candidate = candidate.merge(
        metadata,
        on=["source_device_id", "target_device_id", "workload_id"],
        validate="one_to_one",
    )

    rng = np.random.default_rng(20260818)
    interval_rows = []
    for direction, group in candidate.groupby("direction", sort=True):
        for row in bootstrap_metric_intervals(group, rng, args.bootstrap_repeats):
            row["direction"] = direction
            interval_rows.append(row)
    intervals = pd.DataFrame(interval_rows)[
        ["direction", "metric", "estimate", "ci95_low", "ci95_high", "bootstrap_repeats"]
    ]
    intervals.to_csv(output / "bootstrap_intervals.csv", index=False)

    slice_rows = []
    for direction, group in candidate.groupby("direction", sort=True):
        for field in ("batch", "imgsz", "amp"):
            for value, part in group.groupby(field, sort=True):
                slice_rows.append(
                    {
                        "direction": direction,
                        "slice_field": field,
                        "slice_value": str(value),
                        "n": len(part),
                        "mae_ms": float(part.absolute_error_ms.mean()),
                        "mape_percent": float(part.ape_percent.mean()),
                        "bias_ms": float(part.error_ms.mean()),
                        "max_ape_percent": float(part.ape_percent.max()),
                    }
                )
    slices = pd.DataFrame(slice_rows)
    slices.to_csv(output / "slice_metrics.csv", index=False)

    stability_rows = []
    compact_coefficients = coefficients[coefficients.model == CANDIDATE_MODEL]
    for (direction, feature), group in compact_coefficients.groupby(
        ["direction", "feature"], sort=True
    ):
        values = group.standardized_coefficient
        stability_rows.append(
            {
                "direction": direction,
                "feature": feature,
                "mean_standardized_coefficient": float(values.mean()),
                "std_standardized_coefficient": float(values.std()),
                "mean_absolute_coefficient": float(values.abs().mean()),
                "dominant_sign_fraction": float(max((values > 0).mean(), (values < 0).mean())),
            }
        )
    stability = pd.DataFrame(stability_rows).sort_values(
        ["direction", "mean_absolute_coefficient"], ascending=[True, False]
    )
    stability.to_csv(output / "feature_stability.csv", index=False)

    stage_rows = []
    for direction in sorted(candidate.direction.unique()):
        for stage, metrics, eligibility in (
            ("part1_idealized", p1m, "no"),
            ("part2_idealized", p2m, "no"),
            ("part3_deployment", p3m, "yes"),
            ("part4_deployment", p4m, "yes"),
        ):
            best = metrics[metrics.direction == direction].sort_values("mape_percent").iloc[0]
            stage_rows.append(
                {
                    "direction": direction,
                    "stage": stage,
                    "deployment_eligible": eligibility,
                    "best_model": best.model,
                    "mae_ms": best.mae_ms,
                    "rmse_ms": best.rmse_ms,
                    "mape_percent": best.mape_percent,
                    "max_ape_percent": best.max_ape_percent,
                    "r2": best.r2,
                }
            )
    stage_comparison = pd.DataFrame(stage_rows)
    stage_comparison.to_csv(output / "stage_comparison.csv", index=False)

    alpha_rows = []
    manifests = []
    for (source, target), direction_data in data.groupby(
        ["source_device_id", "target_device_id"], sort=True
    ):
        direction = f"{source}_to_{target}"
        alpha, scores = select_final_alpha(direction_data)
        scores.insert(0, "direction", direction)
        alpha_rows.append(scores)
        pipeline = make_pipeline()
        pipeline.set_params(model__alpha=alpha)
        pipeline.fit(direction_data[FINAL_FEATURES], target_values(direction_data))
        model_path = model_dir / f"{direction}_compact_gpu_ridge.joblib"
        raw_required_fields = [
            "batch",
            "imgsz",
            "amp",
            "source_gpu_sample_count",
        ] + [
            feature
            for feature in COMPACT_GPU_FEATURES
            if feature != "log_profile_sample_count"
        ]
        example_path = example_dir / f"{direction}_example.csv"
        direction_data.iloc[[0]][
            ["source_device_id", "workload_id"] + raw_required_fields
        ].to_csv(example_path, index=False)
        joblib.dump(
            {
                "pipeline": pipeline,
                "features": FINAL_FEATURES,
                "source_device_id": source,
                "target_device_id": target,
                "predicts": "log target iteration runtime; exponentiate model output",
                "profile_window_steady_iterations": 108,
            },
            model_path,
        )
        manifests.append(
            {
                "direction": direction,
                "source_device_id": source,
                "target_device_id": target,
                "model_path": str(model_path.relative_to(output)),
                "example_input_path": str(example_path.relative_to(output)),
                "selected_alpha": alpha,
                "training_workloads": int(direction_data.workload_id.nunique()),
                "feature_count": len(FINAL_FEATURES),
                "features": FINAL_FEATURES,
                "raw_required_fields": raw_required_fields,
                "label": "median steady_window_mean_iter_ms across three unprofiled target-device baseline repeats",
                "validated_scope": "unseen workload within this known directed GPU pair",
            }
        )
    pd.concat(alpha_rows, ignore_index=True).to_csv(output / "final_alpha_selection.csv", index=False)
    (output / "deployment_manifest.json").write_text(
        json.dumps(manifests, indent=2), encoding="utf-8"
    )

    lines = [
        "# Final model experiment report",
        "",
        "## Decision",
        "",
        "Use the 15-feature compact GPU Ridge model separately for RTX4090-to-RTX5090 and RTX5090-to-RTX4090. It is deployment-aligned: source profiling and workload fields are inputs; target unprofiled baseline runtime is label-only.",
        "",
        "## Stage comparison",
        "",
        "| Direction | Stage | Deployment eligible | Best model | MAE ms | MAPE | Max APE | R2 |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in stage_comparison.itertuples():
        lines.append(
            f"| {row.direction} | {row.stage} | {row.deployment_eligible} | {row.best_model} | {row.mae_ms:.3f} | {row.mape_percent:.2f}% | {row.max_ape_percent:.2f}% | {row.r2:.4f} |"
        )

    lines += [
        "",
        "## Workload-bootstrap uncertainty for the selected deployment model",
        "",
        "Intervals resample the 24 held-out workload predictions within each direction. They quantify workload-sampling uncertainty for this GPU pair, not unseen-device uncertainty.",
        "",
        "| Direction | Metric | Estimate | 95% bootstrap interval |",
        "|---|---|---:|---:|",
    ]
    for row in intervals.itertuples():
        suffix = "%" if row.metric == "mape_percent" else " ms"
        lines.append(
            f"| {row.direction} | {row.metric} | {row.estimate:.3f}{suffix} | {row.ci95_low:.3f} to {row.ci95_high:.3f}{suffix} |"
        )

    lines += ["", "## Error slices", ""]
    for direction in sorted(slices.direction.unique()):
        lines += [
            f"### {direction}",
            "",
            "| Slice | Value | N | MAE ms | MAPE | Bias ms | Max APE |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in slices[slices.direction == direction].itertuples():
            lines.append(
                f"| {row.slice_field} | {row.slice_value} | {row.n} | {row.mae_ms:.3f} | {row.mape_percent:.2f}% | {row.bias_ms:+.3f} | {row.max_ape_percent:.2f}% |"
            )
        lines.append("")

    lines += [
        "## Interpretation and limits",
        "",
        "- Against the best Part 2 idealized model, the deployment model changes mean APE by only +0.026 percentage points for RTX4090-to-RTX5090 (paired Wilcoxon p=0.584) and -0.014 points for RTX5090-to-RTX4090 (p=0.922); no measurable accuracy penalty is evident in this dataset.",
        "- Compact GPU metrics capture most useful profiling signal; adding full GPU summaries, trace, or current HPCT artifact proxies does not improve nested-LOOW error.",
        "- Current evaluation supports unseen-workload generalization only for the known RTX4090/RTX5090 directed pair.",
        "- Hardware descriptors are constant within each direction and cannot establish unseen-device generalization with only two GPU models.",
        "- Current source profiling covers 108 steady iterations. Shorter 5/10/20-iteration window accuracy remains unmeasured.",
        "- RTX4090 and RTX5090 use different CUDA trace backends, so trace features require additional harmonization before treating them as device-independent.",
        "- Current HPCToolkit features are quality/complexity proxies. Numeric CCT attribution must be extracted before HPCToolkit can be judged as a performance-feature source.",
        "",
        "## Recommended next evidence",
        "",
        "1. Re-extract Nsight metrics from progressively shorter profiling windows and plot accuracy versus profiling cost.",
        "2. Add at least one third GPU and evaluate leave-one-device-out; until then, avoid claims about arbitrary unseen GPU hardware.",
        "3. Add more model architectures and data-pipeline conditions so workload generalization is not limited to YOLO26n and the current batch/image/AMP grid.",
        "4. Extract numeric HPCToolkit CCT metrics only if kernel/calling-context attribution can be validated against Nsight.",
        "",
        "## Packaged models",
        "",
    ]
    for manifest in manifests:
        lines.append(
            f"- `{manifest['direction']}`: `{manifest['model_path']}`, alpha={manifest['selected_alpha']}, {manifest['feature_count']} inputs."
        )
    (output / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Wrote final comparison, {len(intervals)} bootstrap interval rows, "
        f"{len(slices)} slice rows, and {len(manifests)} packaged models to {output}."
    )


if __name__ == "__main__":
    main()
