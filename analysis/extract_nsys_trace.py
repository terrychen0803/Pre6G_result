from __future__ import annotations

import argparse
import sqlite3
import statistics
from pathlib import Path

from common import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR, percentile, write_csv


BASE = ["device_id","workload_id","trace_available","cuda_trace_implementation","nvtx_collection_warning","nvtx_steady_ranges","steady_start_ns","steady_end_ns","kernel_count","unique_kernel_count","total_kernel_time_ms","mean_kernel_duration_us","median_kernel_duration_us","p90_kernel_duration_us","p95_kernel_duration_us","max_kernel_duration_us","kernel_launch_interval_mean_us","gpu_idle_gap_total_ms","gpu_idle_gap_mean_us","memcpy_count","memcpy_bytes","memcpy_time_ms","memcpy_effective_bandwidth_gbps","memset_count","memset_bytes","memset_time_ms","sync_count","sync_time_ms","cuda_api_count","cuda_api_time_ms","warning"]
TOP = ["device_id","workload_id","rank","kernel_name","count","total_duration_ms","mean_duration_us","percent_kernel_time"]
API = ["device_id","workload_id","api_name","count","total_duration_ms","mean_duration_us","median_duration_us","p90_duration_us","p95_duration_us","max_duration_us"]
MEM = ["device_id","workload_id","copy_kind","count","bytes","duration_ms","effective_bandwidth_gbps"]


def durations(con, table, start, end):
    return [r[0] for r in con.execute(f"select end-start from {table} where start>=? and end<=? order by end-start",(start,end))]


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--cache-dir",type=Path,default=DEFAULT_CACHE_DIR/"nsys"); parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT_DIR); args=parser.parse_args()
    rows=[]; tops=[]; apis=[]; mems=[]
    for path in sorted(args.cache_dir.resolve().glob("*/*/nsys_trace/profile.sqlite")):
        device,workload=path.parts[-4],path.parts[-3]; row={"device_id":device,"workload_id":workload,"warning":""}; con=sqlite3.connect(path)
        try:
            tables={r[0] for r in con.execute("select name from sqlite_master where type='table'")}
            required={"NVTX_EVENTS","CUPTI_ACTIVITY_KIND_KERNEL","StringIds"}
            if not required<=tables: row.update(trace_available=False,warning="required trace tables unavailable"); rows.append(row); continue
            steady=con.execute("select count(*),min(start),max(end) from NVTX_EVENTS where text>='TRAIN_ITER_0021' and text<='TRAIN_ITER_0128'").fetchone(); nrange,start,end=steady
            if not start or not end: row.update(trace_available=False,warning="steady NVTX window unavailable"); rows.append(row); continue
            diagnostic_text=" | ".join(str(value[0]) for value in con.execute("select text from DIAGNOSTIC_EVENT") if value[0]) if "DIAGNOSTIC_EVENT" in tables else ""
            implementation="hardware" if "Hardware tracing used for CUDA tracing" in diagnostic_text else ("legacy_software" if "legacy (software instrumented) trace" in diagnostic_text else "unknown")
            kd=durations(con,"CUPTI_ACTIVITY_KIND_KERNEL",start,end); total_ns=sum(kd)
            starts_ends=con.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=? order by start",(start,end)).fetchall()
            intervals=[starts_ends[i][0]-starts_ends[i-1][0] for i in range(1,len(starts_ends))]
            gaps=[]
            if starts_ends:
                busy_until=starts_ends[0][1]
                for kernel_start,kernel_end in starts_ends[1:]:
                    if kernel_start>busy_until: gaps.append(kernel_start-busy_until)
                    busy_until=max(busy_until,kernel_end)
            row.update(trace_available=True,cuda_trace_implementation=implementation,nvtx_collection_warning="Not all NVTX events might have been collected" in diagnostic_text,nvtx_steady_ranges=nrange,steady_start_ns=start,steady_end_ns=end,kernel_count=len(kd),unique_kernel_count=con.execute("select count(distinct demangledName) from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?",(start,end)).fetchone()[0],total_kernel_time_ms=total_ns/1e6,mean_kernel_duration_us=statistics.fmean(kd)/1e3 if kd else None,median_kernel_duration_us=statistics.median(kd)/1e3 if kd else None,p90_kernel_duration_us=percentile(kd,.9)/1e3 if kd else None,p95_kernel_duration_us=percentile(kd,.95)/1e3 if kd else None,max_kernel_duration_us=max(kd)/1e3 if kd else None,kernel_launch_interval_mean_us=statistics.fmean(intervals)/1e3 if intervals else None,gpu_idle_gap_total_ms=sum(gaps)/1e6,gpu_idle_gap_mean_us=statistics.fmean(gaps)/1e3 if gaps else None)
            top=con.execute("select coalesce(s.value,'<unknown>'),count(*),sum(k.end-k.start),avg(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k left join StringIds s on s.id=k.demangledName where k.start>=? and k.end<=? group by k.demangledName order by 3 desc limit 20",(start,end)).fetchall()
            for rank,(name,count,total,mean) in enumerate(top,1): tops.append({"device_id":device,"workload_id":workload,"rank":rank,"kernel_name":name,"count":count,"total_duration_ms":total/1e6,"mean_duration_us":mean/1e3,"percent_kernel_time":100*total/total_ns if total_ns else None})
            if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
                groups=con.execute("select coalesce(s.value,'<unknown>'),count(*),sum(r.end-r.start),avg(r.end-r.start),max(r.end-r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r left join StringIds s on s.id=r.nameId where r.start>=? and r.end<=? group by r.nameId order by 3 desc",(start,end)).fetchall()
                all_api=[]
                for name,count,total,mean,maxv in groups:
                    vals=[x[0] for x in con.execute("select r.end-r.start from CUPTI_ACTIVITY_KIND_RUNTIME r left join StringIds s on s.id=r.nameId where s.value=? and r.start>=? and r.end<=? order by r.end-r.start",(name,start,end))]
                    all_api.extend(vals); apis.append({"device_id":device,"workload_id":workload,"api_name":name,"count":count,"total_duration_ms":total/1e6,"mean_duration_us":mean/1e3,"median_duration_us":statistics.median(vals)/1e3,"p90_duration_us":percentile(vals,.9)/1e3,"p95_duration_us":percentile(vals,.95)/1e3,"max_duration_us":maxv/1e3})
                row.update(cuda_api_count=len(all_api),cuda_api_time_ms=sum(all_api)/1e6)
            if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
                mgroups=con.execute("select coalesce(e.label,e.name,cast(m.copyKind as text)),count(*),sum(m.bytes),sum(m.end-m.start) from CUPTI_ACTIVITY_KIND_MEMCPY m left join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind where m.start>=? and m.end<=? group by m.copyKind",(start,end)).fetchall()
                for kind,count,byte,total in mgroups: mems.append({"device_id":device,"workload_id":workload,"copy_kind":kind,"count":count,"bytes":byte,"duration_ms":total/1e6,"effective_bandwidth_gbps":byte/total if total else None})
                row.update(memcpy_count=sum(x[1] for x in mgroups),memcpy_bytes=sum(x[2] for x in mgroups),memcpy_time_ms=sum(x[3] for x in mgroups)/1e6)
                row["memcpy_effective_bandwidth_gbps"]=row["memcpy_bytes"]/(row["memcpy_time_ms"]*1e6) if row["memcpy_time_ms"] else None
            for table,prefix in (("CUPTI_ACTIVITY_KIND_MEMSET","memset"),("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION","sync")):
                if table in tables:
                    count,total=con.execute(f"select count(*),coalesce(sum(end-start),0) from {table} where start>=? and end<=?",(start,end)).fetchone(); row[f"{prefix}_count"]=count; row[f"{prefix}_time_ms"]=total/1e6
                    if prefix=="memset": row["memset_bytes"]=con.execute(f"select coalesce(sum(bytes),0) from {table} where start>=? and end<=?",(start,end)).fetchone()[0]
            rows.append(row)
        finally: con.close()
        print(f"{device}/{workload}: extracted",flush=True)
    out=args.output_dir.resolve(); write_csv(out/"nsys_trace_features.csv",rows,BASE); write_csv(out/"nsys_trace_top_kernels.csv",tops,TOP); write_csv(out/"nsys_trace_cuda_api.csv",apis,API); write_csv(out/"nsys_trace_memcpy.csv",mems,MEM); print(f"Extracted trace features for {len(rows)} workloads.")


if __name__=="__main__": main()
