"""Print a compact report summary from evaluation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_MODEL_COMPARISON = Path("artifacts/evaluation/model_comparison.csv")


def format_metric(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6f}"


def load_model_comparison(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Model comparison artifact not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Model comparison artifact is empty: {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a report summary from evaluation artifacts.")
    parser.add_argument("--path", type=Path, default=DEFAULT_MODEL_COMPARISON, help="Path to model_comparison.csv")
    args = parser.parse_args()

    frame = load_model_comparison(args.path)
    metric_columns = [column for column in ["accuracy", "precision", "recall", "f1"] if column in frame.columns]

    print("Evaluation Summary")
    print("===================")
    for _, row in frame.iterrows():
        model_name = str(row.get("model", "unknown"))
        print(f"{model_name}:")
        for column in metric_columns:
            print(f"  {column.capitalize():<9} {format_metric(row.get(column))}")
        print()

    if metric_columns:
        print("Best metric by model:")
        for column in metric_columns:
            series = frame[column].astype(float)
            best_index = series.idxmax()
            best_row = frame.loc[best_index]
            print(f"  {column.capitalize():<9} {best_row['model']} = {format_metric(best_row[column])}")


if __name__ == "__main__":
    main()