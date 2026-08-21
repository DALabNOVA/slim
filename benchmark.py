"""Repeatable end-to-end SLIM benchmark for the two bundled sklearn datasets."""

import argparse
import statistics
import time

import torch
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

from slim_gsgp.main_slim import slim


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(dataset_name, dataset_loader, *, generations=500, repeats=3, device="cpu"):
    device = torch.device(device)
    data = dataset_loader()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.as_tensor(y_test, dtype=torch.float32, device=device)

    timings = []
    for repeat in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _synchronize(device)
        start = time.perf_counter()
        elite = slim(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            dataset_name=dataset_name,
            slim_version="SLIM+SIG1",
            pop_size=100,
            n_iter=generations,
            log_level=0,
            verbose=0,
            seed=42,
        )
        _synchronize(device)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        print(f"{dataset_name} run {repeat + 1}/{repeats}: {elapsed:.3f}s, fitness={elite.fitness:.8g}")

    result = {
        "median_seconds": statistics.median(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
    }
    if device.type == "cuda":
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        result["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
    print(f"{dataset_name}: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"torch={torch.__version__}, device={args.device}, threads={torch.get_num_threads()}")
    run_benchmark("Breast Cancer", load_breast_cancer, generations=args.generations, repeats=args.repeats, device=args.device)
    run_benchmark("Diabetes", load_diabetes, generations=args.generations, repeats=args.repeats, device=args.device)
