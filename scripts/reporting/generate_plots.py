# scripts/reporting/generate_plots.py

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_ROOT = PROJECT_ROOT / "reports"
MASTER_DIR = REPORTS_ROOT / "master"
PLOTS_DIR = REPORTS_ROOT / "plots"

SUMMARY_CSV = MASTER_DIR / "master_summary_table.csv"

BLOCK_ORDER = [
    "1k_short",
    "1k_long",
    "3k_long",
    "5k_long",
    "5k_curriculum_length",
]

METHOD_ORDER = [
    "Random",
    "Grouped",
    "Grouped→Random",
    "Random→Grouped",
    "Easy→Hard Length",
    "Hard→Easy Length",
]


def load_master_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Master summary not found: {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV)

    required = {
        "Block",
        "Method",
        "Mean Eval Loss",
        "Std Eval Loss",
        "Mean Phase2 ΔEval",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Master summary missing columns: {sorted(missing)}")

    return df


def ordered_methods_for_block(df: pd.DataFrame) -> list[str]:
    present = set(df["Method"].tolist())
    ordered = [m for m in METHOD_ORDER if m in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def make_eval_loss_plot(df: pd.DataFrame, block: str) -> None:
    d = df[df["Block"] == block].copy()
    if d.empty:
        return

    order = ordered_methods_for_block(d)
    d = d.set_index("Method").loc[order].reset_index()

    plt.figure(figsize=(10, 5))
    plt.bar(
        d["Method"],
        d["Mean Eval Loss"],
        yerr=d["Std Eval Loss"],
        capsize=5,
    )
    plt.ylabel("Mean Eval Loss")
    plt.title(f"{block}: Mean Eval Loss by Method")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out_path = PLOTS_DIR / f"plot_{block}_eval_loss.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_phase_delta_plot(df: pd.DataFrame, block: str) -> None:
    d = df[df["Block"] == block].copy()
    if d.empty:
        return

    order = ordered_methods_for_block(d)
    d = d.set_index("Method").loc[order].reset_index()

    plt.figure(figsize=(10, 5))
    plt.bar(d["Method"], d["Mean Phase2 ΔEval"])
    plt.ylabel("Mean Phase 2 ΔEval")
    plt.title(f"{block}: Phase 2 Eval Loss Improvement")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out_path = PLOTS_DIR / f"plot_{block}_phase2_delta.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_5k_full_comparison(df: pd.DataFrame) -> None:
    d = df[df["Block"].isin(["5k_long", "5k_curriculum_length"])].copy()
    if d.empty:
        return

    d["Display"] = d["Block"] + " | " + d["Method"]

    display_order = [
        "5k_long | Random",
        "5k_long | Grouped",
        "5k_long | Grouped→Random",
        "5k_long | Random→Grouped",
        "5k_curriculum_length | Easy→Hard Length",
        "5k_curriculum_length | Hard→Easy Length",
    ]
    display_order = [x for x in display_order if x in set(d["Display"])]

    d = d.set_index("Display").loc[display_order].reset_index()

    plt.figure(figsize=(12, 5))
    plt.bar(
        d["Display"],
        d["Mean Eval Loss"],
        yerr=d["Std Eval Loss"],
        capsize=5,
    )
    plt.ylabel("Mean Eval Loss")
    plt.title("5k Comparison: Batching vs Length-Based Curriculum")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    out_path = PLOTS_DIR / "plot_5k_full_curriculum_comparison_eval_loss.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_cross_block_random_trend(df: pd.DataFrame) -> None:
    d = df[df["Method"] == "Random"].copy()
    d = d[d["Block"].isin(["1k_short", "1k_long", "3k_long", "5k_long"])]

    if d.empty:
        return

    d["Block"] = pd.Categorical(d["Block"], categories=BLOCK_ORDER, ordered=True)
    d = d.sort_values("Block")

    plt.figure(figsize=(8, 5))
    plt.plot(d["Block"].astype(str), d["Mean Eval Loss"], marker="o")
    plt.ylabel("Mean Eval Loss")
    plt.title("Random Baseline Eval Loss Across Experiment Blocks")
    plt.tight_layout()

    out_path = PLOTS_DIR / "plot_random_baseline_cross_block_eval_loss.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_master_summary()
    df.to_csv(PLOTS_DIR / "combined_plot_data.csv", index=False)

    for block in BLOCK_ORDER:
        make_eval_loss_plot(df, block)
        make_phase_delta_plot(df, block)

    make_5k_full_comparison(df)
    make_cross_block_random_trend(df)

    print("Created files:")
    for p in sorted(PLOTS_DIR.glob("plot_*.png")):
        print(p.name)
    print("combined_plot_data.csv")


if __name__ == "__main__":
    main()