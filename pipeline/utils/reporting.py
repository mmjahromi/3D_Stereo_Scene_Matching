"""Results table builder and CSV/console reporter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np


class ResultsTable:
    """Accumulate per-model PairResult objects and produce comparison reports."""

    def __init__(self) -> None:
        self._rows: List[dict] = []

    def add(
        self,
        model_name: str,
        T_est: np.ndarray,
        time_s: float,
        num_correspondences: int = 0,
        rre: Optional[float] = None,
        rte: Optional[float] = None,
        success: Optional[bool] = None,
        inlier_ratio: Optional[float] = None,
        chamfer_distance: Optional[float] = None,
    ) -> None:
        self._rows.append(
            dict(
                model=model_name,
                rre=rre,
                rte=rte,
                success=success,
                inlier_ratio=inlier_ratio,
                chamfer=chamfer_distance,
                time_s=time_s,
                num_corr=num_correspondences,
            )
        )

    def _has_gt(self) -> bool:
        return any(r["rre"] is not None for r in self._rows)

    def print_table(self) -> None:
        """Print a formatted comparison table to stdout."""
        from tabulate import tabulate

        headers, rows = self._build_table_data()
        print()
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline", floatfmt=".4f"))
        print()

    def save_csv(self, output_dir: str, filename: str = "results.csv") -> str:
        """Save results to CSV. Returns the output path."""
        import pandas as pd

        os.makedirs(output_dir, exist_ok=True)
        path = str(Path(output_dir) / filename)

        df = pd.DataFrame(self._rows)
        df.to_csv(path, index=False)
        return path

    def recall(self) -> Optional[float]:
        """Fraction of pairs with success=True (if GT available)."""
        successes = [r["success"] for r in self._rows if r["success"] is not None]
        if not successes:
            return None
        return float(np.mean(successes))

    def _build_table_data(self):
        has_gt = self._has_gt()

        if has_gt:
            headers = [
                "Model", "RRE (deg)", "RTE (m)", "Recall",
                "IR", "CD (m)", "Time(s)", "#Corr",
            ]
        else:
            headers = ["Model", "Time(s)", "#Corr"]

        rows = []
        for r in self._rows:
            if has_gt:
                row = [
                    r["model"],
                    r["rre"] if r["rre"] is not None else "—",
                    r["rte"] if r["rte"] is not None else "—",
                    "✓" if r["success"] else "✗" if r["success"] is not None else "—",
                    r["inlier_ratio"] if r["inlier_ratio"] is not None else "—",
                    r["chamfer"] if r["chamfer"] is not None else "—",
                    r["time_s"],
                    r["num_corr"],
                ]
            else:
                row = [r["model"], r["time_s"], r["num_corr"]]
            rows.append(row)

        return headers, rows

    def print_benchmark_summary(self, model_results: dict) -> None:
        """Print a per-model aggregate summary when running multi-pair benchmarks."""
        from tabulate import tabulate

        headers = [
            "Model", "Mean RRE (deg)", "Mean RTE (m)", "Recall",
            "Mean IR", "Mean CD (m)", "Mean Time(s)",
        ]
        rows = []
        for model_name, results in model_results.items():
            rres = [r["rre"] for r in results if r["rre"] is not None]
            rtes = [r["rte"] for r in results if r["rte"] is not None]
            successes = [r["success"] for r in results if r["success"] is not None]
            irs = [r["inlier_ratio"] for r in results if r["inlier_ratio"] is not None]
            cds = [r["chamfer"] for r in results if r["chamfer"] is not None]
            times = [r["time_s"] for r in results]

            rows.append([
                model_name,
                f"{np.mean(rres):.4f}" if rres else "—",
                f"{np.mean(rtes):.4f}" if rtes else "—",
                f"{np.mean(successes):.3f}" if successes else "—",
                f"{np.mean(irs):.4f}" if irs else "—",
                f"{np.mean(cds):.4f}" if cds else "—",
                f"{np.mean(times):.3f}",
            ])

        print()
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
        print()
