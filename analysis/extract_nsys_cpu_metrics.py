from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from common import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR, write_csv


FIELDS=["device_id","workload_id","cpu_pmu_available","candidate_counter_tables","requested_counter_count","requested_counters","diagnostic_warning_count","perf_parse_error","diagnostics"]


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--cache-dir",type=Path,default=DEFAULT_CACHE_DIR/"nsys"); p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT_DIR); args=p.parse_args(); rows=[]
    for path in sorted(args.cache_dir.resolve().glob("*/*/nsys_cpu_metrics/profile.sqlite")):
        device,workload=path.parts[-4],path.parts[-3]; con=sqlite3.connect(path)
        try:
            tables=[r[0] for r in con.execute("select name from sqlite_master where type='table' order by name")]
            # Actual sampled PMU data tables have CPU/perf/sample/counter naming;
            # metadata, enums, diagnostics, and generic CUDA tables are excluded.
            candidates=[t for t in tables if any(x in t.upper() for x in ("CPU_METRIC","PERF_SAMPLE","PMU","CPU_COUNTER"))]
            diagnostics=[]
            if "DIAGNOSTIC_EVENT" in tables: diagnostics=[(int(s),str(t)) for s,t in con.execute("select severity,text from DIAGNOSTIC_EVENT where severity>=2 order by timestamp")]
            requested="cpu-cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses,L1-dcache-loads,LLC-load-misses"
            text=" | ".join(t for _,t in diagnostics)
            rows.append({"device_id":device,"workload_id":workload,"cpu_pmu_available":bool(candidates),"candidate_counter_tables":";".join(candidates),"requested_counter_count":8,"requested_counters":requested,"diagnostic_warning_count":len(diagnostics),"perf_parse_error":"LinuxPerfEventReader" in text,"diagnostics":text})
        finally: con.close()
    write_csv(args.output_dir.resolve()/"nsys_cpu_metrics_availability.csv",rows,FIELDS); print(f"Inspected CPU metric availability for {len(rows)} workloads.")


if __name__=="__main__": main()
