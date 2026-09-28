from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd

from common import DEFAULT_OUTPUT_DIR


def bars(path: Path, labels, values, title: str, ylabel: str, threshold=None) -> None:
    width,height=1100,520; left,right,top,bottom=85,25,55,90; pw=width-left-right; ph=height-top-bottom
    maximum=max(max(values)*1.08,threshold or 0,1); step=pw/max(len(values),1); bw=step*.72
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/>',f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{html.escape(title)}</text>']
    for i in range(6):
        val=maximum*i/5; y=top+ph-(val/maximum)*ph; parts += [f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>',f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{val:.1f}</text>']
    if threshold is not None:
        y=top+ph-(threshold/maximum)*ph; parts += [f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d62728" stroke-dasharray="6 4"/>']
    for i,(label,val) in enumerate(zip(labels,values)):
        x=left+i*step+(step-bw)/2; y=top+ph-(val/maximum)*ph; color="#d95f02" if threshold is not None and val>threshold else "#2878b5"
        parts += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{top+ph-y:.1f}" fill="{color}"/>',f'<text x="{x+bw/2:.1f}" y="{top+ph+18}" text-anchor="end" transform="rotate(-45 {x+bw/2:.1f} {top+ph+18})" font-family="sans-serif" font-size="10">{html.escape(str(label))}</text>']
    parts += [f'<text x="18" y="{top+ph/2}" transform="rotate(-90 18 {top+ph/2})" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(ylabel)}</text>','</svg>']
    path.write_text("\n".join(parts),encoding="utf-8")


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT_DIR); args=p.parse_args(); out=args.output_dir.resolve(); figures=out/"figures"; figures.mkdir(parents=True,exist_ok=True)
    gt=pd.read_csv(out/"ground_truth.csv").sort_values(["device_id","workload_id"])
    combined_labels=[f"{device.replace('RTX','')}-{workload}" for device,workload in zip(gt.device_id,gt.workload_id)]
    bars(figures/"ground_truth_by_workload.svg",combined_labels,gt.ground_truth_iter_ms,"Ground-truth runtime by device and workload","ms / iteration")
    bars(figures/"baseline_cv.svg",combined_labels,gt.baseline_cv*100,"Baseline repeat coefficient of variation","CV (%)",threshold=2)
    for device,group in gt.groupby("device_id"):
        bars(figures/f"ground_truth_{device.lower()}.svg",group.workload_id,group.ground_truth_iter_ms,f"{device} ground-truth runtime by workload","ms / iteration")
        bars(figures/f"baseline_cv_{device.lower()}.svg",group.workload_id,group.baseline_cv*100,f"{device} baseline repeat coefficient of variation","CV (%)",threshold=2)
    cross=gt.pivot(index="workload_id",columns="device_id",values="ground_truth_iter_ms")
    if {"RTX4090","RTX5090"} <= set(cross.columns):
        speedup=cross.RTX4090/cross.RTX5090
        bars(figures/"rtx5090_speedup_over_rtx4090.svg",speedup.index,speedup.values,"RTX5090 speedup over RTX4090","RTX4090 time / RTX5090 time")
    ov=pd.read_csv(out/"profiler_overhead.csv"); summary=ov.groupby(["device_id","profiler_mode"]).overhead_ratio.median().sort_values()
    labels=[f"{device.replace('RTX','')}-{mode.replace('nsys_','')}" for device,mode in summary.index]
    bars(figures/"profiler_overhead_median.svg",labels,summary.values,"Median profiler overhead ratio","profile / baseline")
    print(f"Wrote {2*gt.device_id.nunique()+4} dependency-free SVG figures.")


if __name__=="__main__": main()
