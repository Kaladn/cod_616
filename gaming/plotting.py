"""Deterministic plotting utilities (matplotlib Agg backend)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict
from pathlib import Path

from .session_schema import PLAYBACK_SCHEMA_VERSION


def plot_metric_curve(values: List[float], metric: str, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 2))
    plt.plot(range(len(values)), values, linestyle='-', marker='.', markersize=3)
    plt.title(f"{metric} curve")
    plt.xlabel('index')
    plt.ylabel(metric)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def plot_sector_timeseries(series: Dict[str, List[float]], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 2))
    for k in ("front", "side", "rear"):
        plt.plot(range(len(series.get(k, []))), series.get(k, []), label=k)
    plt.legend()
    plt.title("Sector time series")
    plt.xlabel('index')
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def plot_diff_histogram(deltas: List[float], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 2))
    plt.hist(deltas, bins=min(20, max(1, len(deltas))), color='gray')
    plt.title('Delta histogram')
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
