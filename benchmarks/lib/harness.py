import csv
import json
import os
import subprocess
from typing import cast

import torch
import triton
import triton.testing


def capture_env():
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    try:
        driver = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            )
            .strip()
            .splitlines()[0]
        )
    except Exception:
        driver = "unknown"
    return {
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_mem_gb": round(props.total_memory / 1e9, 2),
        "driver": driver,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "clocks_locked": _clocks_locked(),
    }


def _clocks_locked():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=clocks_throttle_reasons.active",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def bench(fn, warmup=25, rep=100):
    median, p20, p80 = cast(
        list,
        triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8]),
    )
    return {"median_ms": median, "p20_ms": p20, "p80_ms": p80}


def peak_memory(fn):
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def write_results(rows, name, env=None):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(here, "results")
    os.makedirs(results_dir, exist_ok=True)
    env = env or {}

    csv_rows = [{**env, **r} for r in rows]
    csv_path = os.path.join(results_dir, f"{name}.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

    json_path = os.path.join(results_dir, f"{name}.json")
    with open(json_path, "w") as f:
        json.dump({"env": env, "rows": rows}, f, indent=2)

    return csv_path, json_path
