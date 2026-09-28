from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import DEFAULT_OUTPUT_DIR, PROJECT_ROOT


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT_DIR); parser.add_argument("--hardware",type=Path,default=PROJECT_ROOT/"hardware"/"hardware_descriptors.csv"); args=parser.parse_args()
    out=args.output_dir.resolve(); base=pd.read_csv(out/"ground_truth.csv")
    for filename in ("nsys_trace_features.csv","nsys_gpu_metrics_features.csv","nsys_cpu_metrics_availability.csv","hpctoolkit_postprocess_inventory.csv"):
        path=out/filename
        if path.exists(): base=base.merge(pd.read_csv(path),on=["device_id","workload_id"],how="left",validate="one_to_one")
    hardware=pd.read_csv(args.hardware.resolve()).add_prefix("hw_").rename(columns={"hw_device_id":"device_id"})
    base=base.merge(hardware,on="device_id",how="left",validate="many_to_one")
    missing=pd.Series(False,index=base.index)
    availability=pd.DataFrame({
        "trace_features_available":base.get("trace_available",missing).fillna(False).astype(bool),
        "gpu_metrics_features_available":base.get("gpu_metrics_available",missing).fillna(False).astype(bool),
        "cpu_pmu_features_available":base.get("cpu_pmu_available",missing).fillna(False).astype(bool),
    })
    base=pd.concat([base,availability],axis=1)
    base.to_csv(out/"workload_features.csv",index=False)
    schema=[]
    for column in base.columns:
        if column in {"device_id","workload_id"}: role,leak="identifier",False
        elif column=="ground_truth_iter_ms": role,leak="label",False
        elif column.startswith("baseline_") or column in {"ground_truth_valid","warning"}: role,leak="label_audit_exclude_from_X",True
        elif column in {"model","batch","imgsz","amp","nbs"}: role,leak="workload_predictor",False
        elif column.startswith("hw_env_"): role,leak="environment_optional",False
        elif column.startswith("hw_"): role,leak="hardware_predictor",False
        elif column.startswith("hpct_") or column.endswith("_available") or "diagnostic" in column or column.endswith("_ns") or column in {"cuda_trace_implementation","nvtx_collection_warning","nvtx_steady_ranges","gpu_metric_count","gpu_sample_count","requested_counter_count","requested_counters","candidate_counter_tables","perf_parse_error"}: role,leak="quality_control",column=="gpu_sample_count"
        elif column.startswith(("gpu_","kernel_","total_kernel_","cuda_api_","memcpy_","memset_","sync_")): role,leak="source_profile_predictor",False
        else: role,leak="provenance_or_quality_control",False
        schema.append({"column":column,"role":role,"potential_label_leakage":leak})
    pd.DataFrame(schema).to_csv(out/"feature_schema.csv",index=False)
    print(f"Wrote {len(base)} model-oriented workload rows with {len(base.columns)} columns.")


if __name__=="__main__": main()
