"""タイマーユーティリティ: FPFH計算とRANSAC処理の時間計測用."""

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import ParamSpec, TypeVar

from utils.setup_logging import setup_logging

logger = setup_logging(__name__)

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class TimingResult:
    """処理時間の計測結果を格納するデータクラス."""

    name: str
    elapsed_seconds: float

    def __str__(self) -> str:
        return f"{self.name}: {self.elapsed_seconds:.4f} seconds"


@dataclass
class BenchmarkResults:
    """ベンチマーク全体の結果を格納するデータクラス."""

    timings: dict[str, float] = field(default_factory=dict)
    fitness_scores: dict[str, float] = field(default_factory=dict)
    inlier_rmse: dict[str, float] = field(default_factory=dict)

    def add_timing(self, name: str, elapsed: float) -> None:
        """処理時間を追加."""
        self.timings[name] = elapsed

    def add_fitness(self, name: str, fitness: float) -> None:
        """fitness値を追加."""
        self.fitness_scores[name] = fitness

    def add_rmse(self, name: str, rmse: float) -> None:
        """inlier_rmse値を追加."""
        self.inlier_rmse[name] = rmse

    def summary(self) -> str:
        """ベンチマーク結果のサマリーを生成."""
        lines = ["=" * 60, "Benchmark Results Summary", "=" * 60]

        if self.timings:
            lines.append("\n[Processing Time]")
            for name, elapsed in self.timings.items():
                lines.append(f"  {name}: {elapsed:.4f} seconds")

        if self.fitness_scores:
            lines.append("\n[Fitness Scores]")
            for name, fitness in self.fitness_scores.items():
                lines.append(f"  {name}: {fitness:.6f}")

        if self.inlier_rmse:
            lines.append("\n[Inlier RMSE]")
            for name, rmse in self.inlier_rmse.items():
                lines.append(f"  {name}: {rmse:.6f}")

        lines.append("=" * 60)
        return "\n".join(lines)


@contextmanager
def timer(name: str) -> Generator[TimingResult, None, None]:
    """処理時間を計測するコンテキストマネージャ.

    Args:
        name: 計測対象の名前

    Yields:
        TimingResult: 計測結果 (elapsed_secondsは終了後に設定される)

    Example:
        >>> with timer("FPFH calculation") as t:
        ...     # 処理
        >>> print(t.elapsed_seconds)
    """
    result = TimingResult(name=name, elapsed_seconds=0.0)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_seconds = time.perf_counter() - start
        logger.info("%s", result)


def timed(name: str | None = None) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """処理時間を計測するデコレータ.

    Args:
        name: 計測対象の名前 (省略時は関数名を使用)

    Returns:
        デコレータ関数

    Example:
        >>> @timed("RANSAC registration")
        ... def global_registration(...):
        ...     ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            label = name or func.__name__
            with timer(label):
                return func(*args, **kwargs)

        return wrapper

    return decorator
