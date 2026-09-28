from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import DEFAULT_OUTPUT_DIR


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT_DIR); args=p.parse_args(); out=args.output_dir.resolve()
    data=pd.read_csv(out/"workload_features.csv")
    numeric=data.select_dtypes(include="number").copy()
    exclude={"ground_truth_iter_ms","baseline_mean_ms","baseline_median_ms","baseline_rep1_ms","baseline_rep2_ms","baseline_rep3_ms","profile_runtime_ms","baseline_ground_truth_ms","overhead_ratio","overhead_percent","summary_steady_window_mean_ms"}
    rows=[]
    target=numeric.get("ground_truth_iter_ms")
    if target is not None:
        for name in numeric.columns:
            if name in exclude or name.startswith(("baseline_","hpct_")) or numeric[name].nunique(dropna=True)<2: continue
            pair=pd.concat([numeric[name],target],axis=1).dropna()
            if len(pair)<4: continue
            rows.append({"feature":name,"n":len(pair),"pearson":pair.iloc[:,0].corr(pair.iloc[:,1],method="pearson"),"spearman":pair.iloc[:,0].corr(pair.iloc[:,1],method="spearman"),"timing_derived":name.startswith(("kernel_","total_kernel","cuda_api","memcpy_","memset_","sync_","gpu_idle_")) or name in {"gpu_sample_count","steady_start_ns","steady_end_ns"}})
    pd.DataFrame(rows).sort_values("spearman",key=lambda s:s.abs(),ascending=False).to_csv(out/"feature_correlations.csv",index=False)
    print(f"Wrote {len(rows)} univariate correlations; timing-derived features are explicitly flagged.")


if __name__=="__main__": main()
