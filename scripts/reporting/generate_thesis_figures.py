from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import ScalarFormatter
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
THESIS_FIGURES_DIR = REPORTS_DIR / "thesis_figures"
MASTER_SUMMARY_CSV = REPORTS_DIR / "master" / "master_summary_table.csv"
PER_SEED_CSV = REPORTS_DIR / "master" / "master_per_seed_results.csv"
GEN_SUMMARY_A_CSV = REPORTS_DIR / "generation_eval_summaries" / "generation_metrics_summary_table.csv"
GEN_SUMMARY_B_CSV = REPORTS_DIR / "generation_eval_5k" / "generation_metrics_summary_table.csv"
FIGURE_INDEX_MD = THESIS_FIGURES_DIR / "FIGURE_INDEX.md"
README_MD = THESIS_FIGURES_DIR / "README.md"


BLOCK_LABELS = {
    "1k_short": "1k short",
    "1k_long": "1k long",
    "3k_long": "3k",
    "5k_long": "5k main",
    "5k_curriculum_length": "5k length",
    "5k_curriculum_length_longer": "Extended hard-to-easy",
}

BLOCK_ORDER = [
    "1k_short",
    "1k_long",
    "3k_long",
    "5k_long",
    "5k_curriculum_length",
    "5k_curriculum_length_longer",
]

DISPLAY_METHODS = {
    "Random": "Random",
    "Grouped": "Grouped",
    "Grouped->Random": "Grouped -> Random",
    "Random->Grouped": "Random -> Grouped",
    "Easy->Hard Length": "Easy -> Hard",
    "Hard->Easy Length": "Hard -> Easy",
    "Hard->Easy Length Longer": "Hard -> Easy\n(longer)",
}

METHOD_ORDER_BY_BLOCK = {
    "1k_short": ["Random", "Grouped", "Grouped->Random", "Random->Grouped"],
    "1k_long": ["Random", "Grouped", "Grouped->Random", "Random->Grouped"],
    "3k_long": ["Random", "Grouped", "Grouped->Random", "Random->Grouped"],
    "5k_long": ["Random", "Grouped", "Grouped->Random", "Random->Grouped"],
    "5k_curriculum_length": ["Easy->Hard Length", "Hard->Easy Length"],
    "5k_curriculum_length_longer": ["Hard->Easy Length", "Hard->Easy Length Longer"],
}

METHOD_STYLES = {
    "Random": {"facecolor": "0.10", "marker": "o", "hatch": None},
    "Grouped": {"facecolor": "0.45", "marker": "s", "hatch": "//"},
    "Grouped->Random": {"facecolor": "0.70", "marker": "^", "hatch": "xx"},
    "Random->Grouped": {"facecolor": "0.25", "marker": "D", "hatch": ".."},
    "Easy->Hard Length": {"facecolor": "0.60", "marker": "P", "hatch": "\\\\"},
    "Hard->Easy Length": {"facecolor": "0.35", "marker": "X", "hatch": "oo"},
    "Hard->Easy Length Longer": {"facecolor": "0.82", "marker": "*", "hatch": "++"},
}


@dataclass
class FigureRecord:
    number: str
    filename: str
    chapter: str
    placement: str
    caption: str
    source_data: str
    caveat: str


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def normalize_method_name(value: str) -> str:
    return (
        value.replace(chr(8594), "->")
        .replace("â†’", "->")
        .replace("  ", " ")
        .strip()
    )


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_master_summary() -> tuple[list[dict[str, object]], str]:
    rows = load_csv_rows(MASTER_SUMMARY_CSV)
    phase2_column = next(column for column in rows[0] if "Phase2" in column)

    normalized: list[dict[str, object]] = []
    for row in rows:
        method = normalize_method_name(row["Method"])
        if row["Block"] == "5k_curriculum_length_longer":
            method = "Hard->Easy Length Longer"
        normalized.append(
            {
                "Block": row["Block"],
                "Method": method,
                "exp_ids": row["exp_ids"],
                "N": int(row["N"]),
                "Seeds": row["Seeds"],
                "Mean Eval Loss": float(row["Mean Eval Loss"]),
                "Std Eval Loss": float(row["Std Eval Loss"]),
                "Mean Train Loss": float(row["Mean Train Loss"]),
                "Mean Gap": float(row["Mean Gap"]),
                "Mean Phase2 Delta Eval": float(row[phase2_column]),
            }
        )
    return normalized, phase2_column


def load_per_seed_results() -> list[dict[str, object]]:
    rows = load_csv_rows(PER_SEED_CSV)
    normalized: list[dict[str, object]] = []
    for row in rows:
        normalized.append(
            {
                "block": row["block"],
                "exp_id": row["exp_id"],
                "method": normalize_method_name(row["method"]),
                "seed": row["seed"],
                "final_eval_loss": float(row["final_eval_loss"]),
            }
        )
    return normalized


def load_generation_summary() -> list[dict[str, object]]:
    rows = load_csv_rows(GEN_SUMMARY_A_CSV) + load_csv_rows(GEN_SUMMARY_B_CSV)
    result: list[dict[str, object]] = []
    for row in rows:
        exp_id = row["exp_id"]
        method = {
            "exp_026": "Random",
            "exp_027": "Grouped",
            "exp_028": "Grouped->Random",
            "exp_029": "Random->Grouped",
            "exp_030": "Easy->Hard Length",
            "exp_031": "Hard->Easy Length",
        }[exp_id[:7]]
        result.append(
            {
                "exp_id": exp_id,
                "method": method,
                "mean_rouge1": float(row["mean_rouge1"]),
                "std_rouge1": float(row["std_rouge1"]),
                "mean_rouge2": float(row["mean_rouge2"]),
                "std_rouge2": float(row["std_rouge2"]),
                "mean_rougeL": float(row["mean_rougeL"]),
                "std_rougeL": float(row["std_rougeL"]),
                "mean_bertscore_f1": float(row["mean_bertscore_f1"]),
                "std_bertscore_f1": float(row["std_bertscore_f1"]),
            }
        )
    return result


def find_value_string(
    master_rows: list[dict[str, object]],
    block: str,
    method: str,
    field: str,
) -> str:
    for row in master_rows:
        if row["Block"] == block and row["Method"] == method:
            return str(row[field])
    raise KeyError(f"Value not found for block={block}, method={method}, field={field}")


def get_rows_for_block(master_rows: list[dict[str, object]], block: str) -> list[dict[str, object]]:
    if block == "5k_curriculum_length_longer":
        standard = next(
            row
            for row in master_rows
            if row["Block"] == "5k_curriculum_length" and row["Method"] == "Hard->Easy Length"
        )
        longer = next(
            row
            for row in master_rows
            if row["Block"] == "5k_curriculum_length_longer" and row["Method"] == "Hard->Easy Length Longer"
        )
        return [standard, longer]

    rows = [row for row in master_rows if row["Block"] == block]
    order = METHOD_ORDER_BY_BLOCK[block]
    ordered_rows: list[dict[str, object]] = []
    for method in order:
        for row in rows:
            if row["Method"] == method:
                ordered_rows.append(row)
                break
    return ordered_rows


def ensure_output_dir() -> None:
    THESIS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def add_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "white",
    edgecolor: str = "black",
    linewidth: float = 1.5,
    fontsize: float = 12,
    weight: str = "normal",
    text_color: str = "black",
    align: str = "center",
    rounded: bool = True,
    linestyle: str = "-",
) -> None:
    boxstyle = "round,pad=0.02,rounding_size=0.02" if rounded else "square,pad=0.02"
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=boxstyle,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2.0 if align == "center" else x + 0.03 * width,
        y + height / 2.0,
        text,
        ha="center" if align == "center" else "left",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color=text_color,
        linespacing=1.3,
    )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    text: str | None = None,
    text_offset: tuple[float, float] = (0.0, 0.0),
    connectionstyle: str = "arc3,rad=0.0",
    linewidth: float = 1.6,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=linewidth,
        color="black",
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)
    if text:
        midpoint = ((start[0] + end[0]) / 2.0 + text_offset[0], (start[1] + end[1]) / 2.0 + text_offset[1])
        ax.text(midpoint[0], midpoint[1], text, ha="center", va="center", fontsize=10)


def setup_diagram_axis(figsize: tuple[float, float]):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    return fig, ax


def save_figure(fig, filename: str) -> Path:
    path = THESIS_FIGURES_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def verify_png(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected figure was not created: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Figure file is empty: {path}")
    with Image.open(path) as image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Figure has invalid dimensions: {path}")
        extrema = image.convert("L").getextrema()
        if extrema[0] == extrema[1]:
            raise ValueError(f"Figure appears visually empty: {path}")


def annotate_bar_values(ax, bars, labels: list[str], y_offset: float) -> None:
    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )


def annotate_horizontal_values(ax, x_values: list[float], y_values: list[float], labels: list[str]) -> None:
    x_min, x_max = ax.get_xlim()
    offset = (x_max - x_min) * 0.02
    for x_value, y_value, label in zip(x_values, y_values, labels):
        ha = "left" if x_value >= 0 else "right"
        x_text = x_value + offset if x_value >= 0 else x_value - offset
        ax.text(x_text, y_value, label, ha=ha, va="center", fontsize=8)


def format_plot_value(value: float, decimals: int = 5) -> str:
    return f"{value:.{decimals}f}"


def panel_ylim(values: list[float], std_values: list[float] | None = None) -> tuple[float, float]:
    if std_values is None:
        std_values = [0.0 for _ in values]
    lower = min(value - std for value, std in zip(values, std_values))
    upper = max(value + std for value, std in zip(values, std_values))
    margin = max((upper - lower) * 0.20, 0.0002)
    if upper == lower:
        margin = max(abs(upper) * 0.05, 0.0002)
    return lower - margin, upper + margin


def disable_axis_offset(ax, axis: str) -> None:
    if axis in {"x", "both"}:
        formatter_x = ScalarFormatter(useOffset=False)
        formatter_x.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter_x)
    if axis in {"y", "both"}:
        formatter_y = ScalarFormatter(useOffset=False)
        formatter_y.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter_y)


def create_figure_1_1() -> FigureRecord:
    fig, ax = setup_diagram_axis((18, 4.2))
    titles = [
        "Pretrained\nLLMs",
        "Instruction\nFine-Tuning",
        "Mini-Batch\nConstruction",
        "Unknown effect of\nbatch scheduling",
        "Research question:\nDoes batch composition/order\naffect fine-tuning?",
    ]
    x_positions = [0.01, 0.205, 0.40, 0.595, 0.79]
    widths = [0.15, 0.15, 0.15, 0.15, 0.20]
    for index, title in enumerate(titles):
        face = "0.15" if index == len(titles) - 1 else "white"
        text_color = "white" if index == len(titles) - 1 else "black"
        fontsize = 11 if index == len(titles) - 1 else 12
        add_box(
            ax,
            x_positions[index],
            0.28,
            widths[index],
            0.42,
            title,
            facecolor=face,
            text_color=text_color,
            weight="bold",
            fontsize=fontsize,
        )
        if index < len(titles) - 1:
            add_arrow(ax, (x_positions[index] + widths[index], 0.50), (x_positions[index + 1], 0.50))
    path = save_figure(fig, "figure_1_1_research_motivation_flow.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 1.1",
        filename=path.name,
        chapter="Chapter 1",
        placement="Section 1.1, after the motivation paragraph",
        caption="Research motivation showing how this thesis narrows from instruction fine-tuning to the unexplored role of mini-batch scheduling.",
        source_data="Schematic based on the thesis setup.",
        caveat="Conceptual figure only; not derived from result data.",
    )


def create_figure_1_2() -> FigureRecord:
    fig, ax = setup_diagram_axis((14, 7))
    add_box(ax, 0.05, 0.12, 0.40, 0.76, "", facecolor="white", linewidth=1.8)
    add_box(ax, 0.55, 0.12, 0.40, 0.76, "", facecolor="white", linewidth=1.8)
    ax.text(0.25, 0.84, "Included", ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.75, 0.84, "Excluded", ha="center", va="center", fontsize=16, weight="bold")
    included = [
        "Supervised instruction fine-tuning",
        "Batch scheduling",
        "Semantic grouping",
        "Curriculum ordering",
        "LoRA fine-tuning",
        "Dolly subsets",
        "Optimization and generation evaluation",
    ]
    excluded = [
        "RLHF",
        "Inference-time RAG",
        "Full model pretraining",
        "Human preference evaluation",
        "Large-scale production deployment",
    ]
    ax.text(0.10, 0.72, "\n".join(f"- {item}" for item in included), ha="left", va="top", fontsize=13, linespacing=1.5)
    ax.text(0.60, 0.72, "\n".join(f"- {item}" for item in excluded), ha="left", va="top", fontsize=13, linespacing=1.5)
    path = save_figure(fig, "figure_1_2_research_scope_overview.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 1.2",
        filename=path.name,
        chapter="Chapter 1",
        placement="Section 1.2, after the scope definition",
        caption="Scope of the thesis, separating the investigated components from areas outside the study.",
        source_data="Schematic based on the thesis scope.",
        caveat="Conceptual figure only; not derived from result data.",
    )


def create_figure_2_1() -> FigureRecord:
    fig, ax = setup_diagram_axis((14, 8))
    outer = Rectangle((0.06, 0.28), 0.88, 0.60, linewidth=1.8, edgecolor="black", facecolor="none", linestyle="--")
    ax.add_patch(outer)
    ax.text(0.50, 0.90, "Existing Work Context", ha="center", va="center", fontsize=16, weight="bold")
    labels = [
        "Instruction\nfine-tuning",
        "Parameter-efficient\nfine-tuning",
        "Data selection\nand curation",
        "Curriculum\nlearning",
        "Semantic embeddings\nand retrieval",
        "Evaluation\nmetrics",
    ]
    positions = [(0.10, 0.63), (0.37, 0.63), (0.64, 0.63), (0.10, 0.38), (0.37, 0.38), (0.64, 0.38)]
    for (x, y), label in zip(positions, labels):
        add_box(ax, x, y, 0.20, 0.16, label)
    add_box(
        ax,
        0.22,
        0.05,
        0.56,
        0.14,
        "Gap:\nMini-batch construction and scheduling\nduring instruction fine-tuning",
        facecolor="0.15",
        text_color="white",
        weight="bold",
    )
    add_arrow(ax, (0.50, 0.28), (0.50, 0.19))
    path = save_figure(fig, "figure_2_1_position_in_existing_work.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 2.1",
        filename=path.name,
        chapter="Chapter 2",
        placement="Section 2.x, after the literature positioning discussion",
        caption="Positioning of the present study relative to existing work in instruction fine-tuning, curriculum learning, PEFT, and semantic retrieval.",
        source_data="Schematic based on the literature-review framing.",
        caveat="Conceptual figure only; not derived from result data.",
    )


def create_figure_3_1() -> FigureRecord:
    fig, ax = setup_diagram_axis((17, 4.2))
    labels = [
        "Dolly\ndataset",
        "Preprocessing",
        "Embedding/index\nconstruction\n(where applicable)",
        "Batch scheduling\nstrategy",
        "LoRA\nfine-tuning",
        "Multi-seed\nevaluation",
        "Result\nanalysis",
    ]
    x = 0.02
    for index, label in enumerate(labels):
        add_box(ax, x, 0.28, 0.11, 0.44, label, weight="bold" if index in {0, 4, 6} else "normal")
        if index < len(labels) - 1:
            add_arrow(ax, (x + 0.11, 0.50), (x + 0.13, 0.50))
        x += 0.13
    path = save_figure(fig, "figure_3_1_methodology_workflow.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 3.1",
        filename=path.name,
        chapter="Chapter 3",
        placement="Section 3.1, methodology overview",
        caption="Overall methodology workflow from dataset preparation through batch scheduling, LoRA fine-tuning, multi-seed evaluation, and result analysis.",
        source_data="Schematic based on `configs/experiments/*.yaml`, `src/batching/*.py`, and `src/training/*.py`.",
        caveat="Conceptual figure only; not derived from result data.",
    )


def create_figure_3_2() -> FigureRecord:
    fig, ax = setup_diagram_axis((14, 7))
    families = [
        ("Static", ["Random", "Grouped"]),
        ("Two-phase", ["Random -> Grouped", "Grouped -> Random"]),
        ("Length-based", ["Easy -> Hard", "Hard -> Easy"]),
    ]
    x_positions = [0.05, 0.36, 0.67]
    for x, (title, items) in zip(x_positions, families):
        add_box(ax, x, 0.14, 0.26, 0.72, "", linewidth=1.8)
        ax.text(x + 0.13, 0.80, title, ha="center", va="center", fontsize=15, weight="bold")
        ax.text(x + 0.04, 0.64, "\n".join(f"- {item}" for item in items), ha="left", va="top", fontsize=13, linespacing=1.8)
    path = save_figure(fig, "figure_3_2_batch_scheduling_strategies.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 3.2",
        filename=path.name,
        chapter="Chapter 3",
        placement="Section 3.2, after the strategy-family description",
        caption="Production strategy families evaluated in the thesis: static batching, two-phase curricula, and length-based curricula.",
        source_data="Schematic based on the final production strategy set.",
        caveat="Mixed batching appears only in exploratory notebooks and is intentionally omitted here.",
    )


def create_figure_3_3() -> FigureRecord:
    fig, ax = setup_diagram_axis((18, 4.4))
    labels = [
        "Instruction\ntext",
        "all-MiniLM-\nL6-v2",
        "Normalized\nvectors",
        "FAISS\nIndexFlatIP",
        "Nearest\nneighbors",
        "Anchor/group\nmapping",
        "Grouped\nmini-batches",
    ]
    x = 0.01
    for index, label in enumerate(labels):
        add_box(ax, x, 0.30, 0.11, 0.40, label)
        if index < len(labels) - 1:
            add_arrow(ax, (x + 0.11, 0.50), (x + 0.125, 0.50))
        x += 0.125
    ax.text(0.50, 0.10, "Notes: top_k = 32 for 1k, top_k = 8 for 3k/5k, batch size = 8", ha="center", va="center", fontsize=12)
    path = save_figure(fig, "figure_3_3_semantic_minibatch_construction.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 3.3",
        filename=path.name,
        chapter="Chapter 3",
        placement="Section 3.3, semantic grouping subsection",
        caption="Schematic of semantic mini-batch construction using sentence embeddings, FAISS nearest-neighbor retrieval, and anchor-based grouping.",
        source_data="Schematic based on `src/batching/grouping.py`, `src/batching/index_loader.py`, and the indexing scripts.",
        caveat="The 1k semantic index differs from 3k/5k in effective `top_k`.",
    )


def create_figure_3_4() -> FigureRecord:
    fig, ax = setup_diagram_axis((14, 8))
    panels = [
        (0.08, 0.55, "Random -> Grouped", "exploration -> refinement"),
        (0.54, 0.55, "Grouped -> Random", "structure -> randomization"),
        (0.08, 0.12, "Easy -> Hard", "shorter / easier first"),
        (0.54, 0.12, "Hard -> Easy", "longer / harder first"),
    ]
    for x, y, title, subtitle in panels:
        add_box(ax, x, y, 0.34, 0.24, "", linewidth=1.8)
        ax.text(x + 0.17, y + 0.16, title, ha="center", va="center", fontsize=15, weight="bold")
        ax.text(x + 0.17, y + 0.08, subtitle, ha="center", va="center", fontsize=12)
    path = save_figure(fig, "figure_3_4_curriculum_scheduling_designs.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 3.4",
        filename=path.name,
        chapter="Chapter 3",
        placement="Section 3.4, curriculum-design subsection",
        caption="Curriculum scheduling designs evaluated in the study, spanning semantic ordering and length-based ordering.",
        source_data="Schematic based on the final production strategy set.",
        caveat="The length-based labels refer to ordering heuristics, not validated difficulty labels.",
    )


def create_figure_4_1() -> FigureRecord:
    fig, ax = setup_diagram_axis((18, 7))
    top_labels = ["Dolly\nsubsets", "Tokenization", "Processed\ndatasets", "Optional\nsemantic index", "Samplers"]
    bottom_labels = ["Two-phase LoRA\ntraining", "Run\nsummaries", "Master\naggregation", "Generation\nevaluation"]
    top_x = [0.03, 0.22, 0.41, 0.60, 0.79]
    for x, label in zip(top_x, top_labels):
        add_box(ax, x, 0.60, 0.16, 0.20, label)
    for start_x, end_x in zip(top_x[:-1], top_x[1:]):
        add_arrow(ax, (start_x + 0.16, 0.70), (end_x, 0.70))
    bottom_x = [0.08, 0.32, 0.56, 0.80]
    for x, label in zip(bottom_x, bottom_labels):
        add_box(ax, x, 0.18, 0.16, 0.20, label)
    add_arrow(ax, (0.87, 0.60), (0.16, 0.38), connectionstyle="angle3,angleA=-90,angleB=180")
    for start_x, end_x in zip(bottom_x[:-1], bottom_x[1:]):
        add_arrow(ax, (start_x + 0.16, 0.28), (end_x, 0.28))
    path = save_figure(fig, "figure_4_1_experimental_pipeline.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 4.1",
        filename=path.name,
        chapter="Chapter 4",
        placement="Section 4.1, experimental setup overview",
        caption="Implementation-specific experimental pipeline from Dolly subset preparation through tokenization, sampling, training, aggregation, and generation evaluation.",
        source_data="Schematic based on `manifests/project_manifest.json`, `scripts/training/*.py`, `scripts/reporting/aggregate_results.py`, and `scripts/evaluation/evaluate_generation_quality.py`.",
        caveat="Conceptual implementation diagram; not derived from endpoint result values.",
    )


def create_figure_4_2() -> FigureRecord:
    fig, ax = setup_diagram_axis((18, 4.6))
    labels = [
        "Raw\nrows",
        "Embedding\ntext",
        "FAISS neighbor\ngraph",
        "Raw-to-processed\nalignment",
        "GroupedBatchSampler",
        "Seen-set\nhandling",
        "Final grouped\nbatches",
    ]
    x = 0.01
    for index, label in enumerate(labels):
        add_box(ax, x, 0.28, 0.11, 0.42, label)
        if index < len(labels) - 1:
            add_arrow(ax, (x + 0.11, 0.49), (x + 0.125, 0.49))
        x += 0.125
    path = save_figure(fig, "figure_4_2_semantic_grouping_pipeline.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 4.2",
        filename=path.name,
        chapter="Chapter 4",
        placement="Section 4.4, semantic grouping implementation subsection",
        caption="Implementation-specific semantic grouping pipeline from raw examples through FAISS retrieval, alignment, sampler logic, and grouped mini-batch construction.",
        source_data="Schematic based on `src/batching/grouping.py`, `src/batching/index_loader.py`, `src/data/alignment.py`, and the indexing scripts.",
        caveat="The 1k semantic index uses older metadata and a different effective neighbor width.",
    )


def create_figure_4_3() -> FigureRecord:
    fig, ax = setup_diagram_axis((16, 4.2))
    add_box(ax, 0.03, 0.24, 0.24, 0.52, "Phase 1\nsampler mode\nLR = 7e-05\nLoRA training", weight="bold")
    add_box(ax, 0.37, 0.34, 0.18, 0.32, "Adapter\nhandoff", facecolor="0.15", text_color="white", weight="bold")
    add_box(ax, 0.65, 0.24, 0.24, 0.52, "Phase 2\nsampler mode\nLR = 5e-05\ncontinued LoRA training", weight="bold")
    add_box(ax, 0.92, 0.34, 0.05, 0.32, "Final\nadapter /\ncheckpoint", linewidth=1.8)
    add_arrow(ax, (0.27, 0.50), (0.37, 0.50))
    add_arrow(ax, (0.55, 0.50), (0.65, 0.50))
    add_arrow(ax, (0.89, 0.50), (0.92, 0.50))
    path = save_figure(fig, "figure_4_3_two_phase_training_flow.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 4.3",
        filename=path.name,
        chapter="Chapter 4",
        placement="Section 4.3, training procedure subsection",
        caption="Two-phase training flow showing phase-specific sampler modes, learning rates, adapter handoff, and the final checkpoint output.",
        source_data="Schematic based on `src/training/experiment_runner.py`, `src/training/phase_runner.py`, and `src/training/trainer_factory.py`.",
        caveat="Conceptual implementation diagram; not derived from endpoint result values.",
    )


def create_figure_4_4() -> FigureRecord:
    fig, ax = setup_diagram_axis((14, 8))
    add_box(ax, 0.24, 0.83, 0.52, 0.10, "Production-implemented batch scheduling strategies", facecolor="0.15", text_color="white", weight="bold")
    families = [
        ("Static", ["Random", "Grouped"], 0.05),
        ("Two-phase", ["Grouped -> Random", "Random -> Grouped"], 0.37),
        ("Length-based", ["Easy -> Hard Length", "Hard -> Easy Length"], 0.69),
    ]
    for title, items, x in families:
        add_box(ax, x, 0.25, 0.24, 0.48, "", linewidth=1.8)
        ax.text(x + 0.12, 0.67, title, ha="center", va="center", fontsize=14, weight="bold")
        ax.text(x + 0.03, 0.56, "\n".join(f"- {item}" for item in items), ha="left", va="top", fontsize=12, linespacing=1.7)
        add_arrow(ax, (0.50, 0.83), (x + 0.12, 0.73))
    ax.text(0.50, 0.10, "Note: mixed batching was exploratory only and is not part of the final production comparison.", ha="center", va="center", fontsize=11)
    path = save_figure(fig, "figure_4_4_batch_scheduling_taxonomy.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 4.4",
        filename=path.name,
        chapter="Chapter 4",
        placement="Section 4.5, strategy summary subsection",
        caption="Taxonomy of the production-implemented batch scheduling strategies used in the final experiment matrix.",
        source_data="Schematic based on the final production strategy set and `src/batching/*.py`.",
        caveat="Mixed batching appears only in exploratory notebooks and should not be presented as part of the final production comparison.",
    )


def create_figure_5_1(master_rows: list[dict[str, object]]) -> FigureRecord:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, block in zip(axes.flatten(), BLOCK_ORDER):
        rows = get_rows_for_block(master_rows, block)
        values = [float(row["Mean Eval Loss"]) for row in rows]
        stds = [float(row["Std Eval Loss"]) for row in rows]
        methods = [DISPLAY_METHODS[row["Method"]] for row in rows]
        styles = [METHOD_STYLES[row["Method"]] for row in rows]
        x_positions = list(range(len(rows)))
        bars = ax.bar(
            x_positions,
            values,
            color=[style["facecolor"] for style in styles],
            edgecolor="black",
            linewidth=1.2,
            yerr=stds,
            capsize=4,
        )
        for bar, style in zip(bars, styles):
            if style["hatch"]:
                bar.set_hatch(style["hatch"])
        ax.set_xticks(x_positions, methods, rotation=20, ha="right")
        ax.set_ylim(*panel_ylim(values, stds))
        ax.set_title(BLOCK_LABELS[block])
        disable_axis_offset(ax, "y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="0.85", linewidth=0.8)
        exact_labels = [str(row["Mean Eval Loss"]) for row in rows]
        y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
        annotate_bar_values(ax, bars, exact_labels, y_offset)
        ax.set_ylabel("Mean eval loss")
    fig.suptitle("Figure 5.1  Mean final evaluation loss across experiment blocks", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = save_figure(fig, "figure_5_1_mean_eval_loss_by_block.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 5.1",
        filename=path.name,
        chapter="Chapter 5",
        placement="Section 5.2, after the optimization summary tables",
        caption="Mean final evaluation loss across scheduling strategies and experiment blocks. Lower is better.",
        source_data=str(MASTER_SUMMARY_CSV),
        caveat="Small visual differences should not be overinterpreted.",
    )


def create_figure_5_2(master_rows: list[dict[str, object]]) -> FigureRecord:
    comparable_blocks = ["1k_short", "1k_long", "3k_long", "5k_long"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, block in zip(axes.flatten(), comparable_blocks):
        rows = get_rows_for_block(master_rows, block)
        random_value = next(float(row["Mean Eval Loss"]) for row in rows if row["Method"] == "Random")
        compare_rows = [row for row in rows if row["Method"] != "Random"]
        y_positions = list(range(len(compare_rows)))
        diffs = [float(row["Mean Eval Loss"]) - random_value for row in compare_rows]
        labels = [DISPLAY_METHODS[row["Method"]] for row in compare_rows]
        colors = [METHOD_STYLES[row["Method"]]["facecolor"] for row in compare_rows]
        bars = ax.barh(y_positions, diffs, color=colors, edgecolor="black", linewidth=1.2)
        for bar, row in zip(bars, compare_rows):
            hatch = METHOD_STYLES[row["Method"]]["hatch"]
            if hatch:
                bar.set_hatch(hatch)
        ax.axvline(0.0, color="black", linewidth=1.4)
        spread = max(max(abs(value) for value in diffs), 0.00015)
        ax.set_xlim(-spread * 1.4, spread * 1.4)
        ax.set_yticks(y_positions, labels)
        ax.invert_yaxis()
        ax.set_title(BLOCK_LABELS[block])
        disable_axis_offset(ax, "x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="0.85", linewidth=0.8)
        annotate_horizontal_values(ax, diffs, y_positions, [f"{value:+.15f}" for value in diffs])
        ax.set_xlabel("Eval loss minus Random")
    fig.suptitle("Figure 5.2  Difference from the Random baseline", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = save_figure(fig, "figure_5_2_difference_from_random_baseline.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 5.2",
        filename=path.name,
        chapter="Chapter 5",
        placement="Section 5.3, random-baseline comparison subsection",
        caption="Difference in mean evaluation loss relative to the Random baseline. Negative values indicate improvement over Random.",
        source_data=str(MASTER_SUMMARY_CSV),
        caveat="All comparable blocks use a centered zero line; most differences are small.",
    )


def create_figure_5_3(master_rows: list[dict[str, object]]) -> FigureRecord:
    comparable_blocks = ["1k_short", "1k_long", "3k_long", "5k_long"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, block in zip(axes.flatten(), comparable_blocks):
        rows = get_rows_for_block(master_rows, block)
        grouped_to_random = next(row for row in rows if row["Method"] == "Grouped->Random")
        random_to_grouped = next(row for row in rows if row["Method"] == "Random->Grouped")
        values = [float(grouped_to_random["Mean Eval Loss"]), float(random_to_grouped["Mean Eval Loss"])]
        minimum, maximum = panel_ylim(values)
        y = 0.5
        ax.plot(values, [y, y], color="black", linewidth=1.2)
        ax.scatter(values[0], y, s=90, marker=METHOD_STYLES["Grouped->Random"]["marker"], color=METHOD_STYLES["Grouped->Random"]["facecolor"], edgecolors="black", zorder=3)
        ax.scatter(values[1], y, s=90, marker=METHOD_STYLES["Random->Grouped"]["marker"], color=METHOD_STYLES["Random->Grouped"]["facecolor"], edgecolors="black", zorder=3)
        better = "Random -> Grouped" if values[1] < values[0] else "Grouped -> Random"
        ax.text(values[0], y + 0.10, f"G->R\n{values[0]:.15f}", ha="center", va="bottom", fontsize=8)
        ax.text(values[1], y - 0.10, f"R->G\n{values[1]:.15f}", ha="center", va="top", fontsize=8)
        ax.text((values[0] + values[1]) / 2.0, 0.88, f"Better: {better}", ha="center", va="center", fontsize=10, weight="bold")
        ax.set_xlim(minimum, maximum)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_title(BLOCK_LABELS[block])
        disable_axis_offset(ax, "x")
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="0.85", linewidth=0.8)
        ax.set_xlabel("Mean eval loss")
    fig.suptitle("Figure 5.3  Two-phase curriculum ordering comparison", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = save_figure(fig, "figure_5_3_curriculum_ordering_comparison.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 5.3",
        filename=path.name,
        chapter="Chapter 5",
        placement="Section 5.4, curriculum-ordering subsection",
        caption="Comparison of two-phase semantic curriculum orderings across comparable experiment blocks.",
        source_data=str(MASTER_SUMMARY_CSV),
        caveat="The axis is zoomed in each panel because the ordering differences are small.",
    )


def create_figure_5_4(master_rows: list[dict[str, object]]) -> FigureRecord:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, block in zip(axes.flatten(), BLOCK_ORDER):
        rows = get_rows_for_block(master_rows, block)
        values = [float(row["Mean Phase2 Delta Eval"]) for row in rows]
        labels = [DISPLAY_METHODS[row["Method"]] for row in rows]
        y_positions = list(range(len(rows)))
        ax.axvline(0.0, color="black", linewidth=1.2)
        for y_position, row, value in zip(y_positions, rows, values):
            style = METHOD_STYLES[row["Method"]]
            ax.scatter(value, y_position, s=90, marker=style["marker"], color=style["facecolor"], edgecolors="black", zorder=3)
            ax.text(
                value,
                y_position + 0.18,
                format_plot_value(value, decimals=5),
                ha="center",
                va="bottom",
                fontsize=7,
            )
        spread = max(abs(min(values)), abs(max(values)))
        ax.set_xlim(-spread * 1.35, spread * 0.25 if spread > 0 else 0.01)
        ax.set_yticks(y_positions, labels)
        ax.invert_yaxis()
        ax.set_title(BLOCK_LABELS[block])
        disable_axis_offset(ax, "x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="0.85", linewidth=0.8)
        ax.set_xlabel("Mean phase-2 delta eval")
    fig.suptitle("Figure 5.4  Phase-2 evaluation-loss change", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = save_figure(fig, "figure_5_4_phase2_delta_eval_comparison.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 5.4",
        filename=path.name,
        chapter="Chapter 5",
        placement="Section 5.2 or 5.4, phase-wise analysis subsection",
        caption="Mean evaluation-loss change during Phase 2. More negative values indicate larger phase-2 improvement.",
        source_data=str(MASTER_SUMMARY_CSV),
        caveat="Cross-block magnitudes differ sharply, so each panel uses its own zoomed range.",
    )


def create_figure_5_5(master_rows: list[dict[str, object]]) -> FigureRecord:
    same_budget_rows = get_rows_for_block(master_rows, "5k_curriculum_length")
    extension_row = get_rows_for_block(master_rows, "5k_curriculum_length_longer")[1]
    rows = same_budget_rows + [extension_row]

    fig, ax = plt.subplots(figsize=(12, 7))
    x_positions = [0.0, 1.0, 3.0]
    values = [float(row["Mean Eval Loss"]) for row in rows]
    labels = [
        "Easy -> Hard\n(same budget)",
        "Hard -> Easy\n(same budget)",
        "Hard -> Easy\n(longer training)",
    ]

    ax.plot(
        x_positions[:2],
        values[:2],
        color="0.35",
        linewidth=1.2,
        linestyle="--",
        zorder=1,
    )

    for x_value, row, value in zip(x_positions, rows, values):
        style = METHOD_STYLES[row["Method"]]
        marker_size = 120 if row["Method"] == "Hard->Easy Length Longer" else 100
        ax.scatter(
            x_value,
            value,
            s=marker_size,
            marker=style["marker"],
            color=style["facecolor"],
            edgecolors="black",
            linewidths=1.0,
            zorder=3,
        )
        y_offset = 0.035 if row["Method"] == "Hard->Easy Length Longer" else 0.025
        ax.text(
            x_value,
            value + y_offset,
            format_plot_value(value, decimals=5),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.axvline(2.0, color="0.55", linewidth=1.0, linestyle=":")
    ax.text(0.5, 11.44, "Same-budget comparison", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(3.0, 11.44, "Longer-training extension", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(
        3.0,
        11.405,
        "exp_032 uses longer training;\nnot same-budget comparable",
        ha="center",
        va="top",
        fontsize=10,
    )

    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(10.85, 11.48)
    ax.set_xticks(x_positions, labels)
    ax.set_ylabel("Mean eval loss")
    ax.set_title("Figure 5.5  Length-based curricula and the longer-training extension")
    disable_axis_offset(ax, "y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.85", linewidth=0.8)
    fig.tight_layout()
    path = save_figure(fig, "figure_5_5_length_curriculum_extended_training.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 5.5",
        filename=path.name,
        chapter="Chapter 5",
        placement="Section 5.5, length-based curriculum subsection",
        caption="Same-budget comparison of length-based curriculum strategies, with the longer-training Hard -> Easy extension shown separately.",
        source_data=str(MASTER_SUMMARY_CSV),
        caveat="The longer-training result is not directly comparable under the same training budget.",
    )


def create_figure_5_6(generation_rows: list[dict[str, object]]) -> FigureRecord:
    metric_specs = [
        ("mean_rouge1", "std_rouge1", "ROUGE-1"),
        ("mean_rouge2", "std_rouge2", "ROUGE-2"),
        ("mean_rougeL", "std_rougeL", "ROUGE-L"),
        ("mean_bertscore_f1", "std_bertscore_f1", "BERTScore F1"),
    ]
    ordered_methods = ["Random", "Grouped", "Grouped->Random", "Random->Grouped", "Easy->Hard Length", "Hard->Easy Length"]
    row_lookup = {row["method"]: row for row in generation_rows}
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    for ax, (mean_key, std_key, title) in zip(axes.flatten(), metric_specs):
        rows = [row_lookup[method] for method in ordered_methods]
        y_positions = list(range(len(rows)))
        values = [float(row[mean_key]) for row in rows]
        stds = [float(row[std_key]) for row in rows]
        ax.errorbar(values, y_positions, xerr=stds, fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=1)
        for y_position, row, value in zip(y_positions, rows, values):
            style = METHOD_STYLES[row["method"]]
            ax.scatter(value, y_position, s=90, marker=style["marker"], color=style["facecolor"], edgecolors="black", zorder=3)
            ax.text(
                value,
                y_position + 0.18,
                format_plot_value(value, decimals=5),
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.set_yticks(y_positions, [DISPLAY_METHODS[row["method"]] for row in rows])
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="0.85", linewidth=0.8)
        metric_min = min(value - std for value, std in zip(values, stds))
        metric_max = max(value + std for value, std in zip(values, stds))
        margin = max((metric_max - metric_min) * 0.25, 0.0002)
        ax.set_xlim(metric_min - margin, metric_max + margin)
        disable_axis_offset(ax, "x")
        ax.set_title(title)
        ax.set_xlabel("Score")
    fig.suptitle("Figure 5.6  Generation-quality metrics for 5k experiments", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = save_figure(fig, "figure_5_6_generation_metrics_comparison.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 5.6",
        filename=path.name,
        chapter="Chapter 5",
        placement="Section 5.6, generation-quality subsection",
        caption="Generation-quality metrics for 5k batching and length-curriculum experiments, plotted from the aggregated generation summary CSVs.",
        source_data=f"{GEN_SUMMARY_A_CSV}; {GEN_SUMMARY_B_CSV}",
        caveat="Aggregated generation CSVs conflict with some per-seed JSON files; use the aggregated CSVs consistently.",
    )


def create_figure_5_7(master_rows: list[dict[str, object]], per_seed_rows: list[dict[str, object]]) -> FigureRecord:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    seed_offsets = {"13": -0.12, "21": 0.00, "42": 0.12}
    for ax, block in zip(axes.flatten(), BLOCK_ORDER):
        rows = get_rows_for_block(master_rows, block)
        methods = [row["Method"] for row in rows]
        display_labels = [DISPLAY_METHODS[method] for method in methods]
        values = [float(row["Mean Eval Loss"]) for row in rows]
        y_min, y_max = panel_ylim(values, [float(row["Std Eval Loss"]) for row in rows])
        for x_index, method in enumerate(methods):
            seed_rows = [
                row
                for row in per_seed_rows
                if row["block"] == block and normalize_method_name(str(row["method"])) == method.replace(" Longer", "")
            ]
            if block == "5k_curriculum_length_longer" and method == "Hard->Easy Length":
                seed_rows = [
                    row
                    for row in per_seed_rows
                    if row["exp_id"] == "exp_031_hard_to_easy_length_multiseed_5k"
                ]
            if method == "Hard->Easy Length Longer":
                seed_rows = [row for row in per_seed_rows if row["exp_id"] == "exp_032_hard_to_easy_length_longer_training_5k"]
            for seed_row in seed_rows:
                seed = str(seed_row["seed"])
                x_value = x_index + seed_offsets.get(seed, 0.0)
                style_key = method if method in METHOD_STYLES else "Hard->Easy Length"
                style = METHOD_STYLES[style_key]
                ax.scatter(
                    x_value,
                    float(seed_row["final_eval_loss"]),
                    s=45,
                    marker=style["marker"],
                    color="white",
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=3,
                )
            mean_row = next(row for row in rows if row["Method"] == method)
            ax.scatter(
                x_index,
                float(mean_row["Mean Eval Loss"]),
                s=95,
                marker="D",
                color=METHOD_STYLES[method]["facecolor"],
                edgecolors="black",
                linewidths=1.0,
                zorder=4,
            )
        ax.set_xticks(range(len(display_labels)), display_labels, rotation=20, ha="right")
        ax.set_ylim(y_min, y_max)
        disable_axis_offset(ax, "y")
        ax.set_title(BLOCK_LABELS[block])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="0.85", linewidth=0.8)
        ax.set_ylabel("Final eval loss")
    fig.suptitle("Figure 5.7  Seed-level final evaluation loss", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = save_figure(fig, "figure_5_7_seed_variance_plot.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 5.7",
        filename=path.name,
        chapter="Chapter 5",
        placement="Section 5.7, stability and seed-variance subsection",
        caption="Per-seed final evaluation loss showing seed-level variability across scheduling strategies; each experiment includes three seeds.",
        source_data=f"{PER_SEED_CSV}; {MASTER_SUMMARY_CSV}",
        caveat="Only three seeds were used per experiment.",
    )


def write_figure_index(records: list[FigureRecord]) -> None:
    lines = [
        "# Figure Index",
        "",
        "This directory contains regenerated thesis-grade figures built from the canonical source files or from explicit methodology schematics.",
        "",
        "## Notes",
        "",
        "- Do not use old plots in `reports/plots/` directly in the thesis.",
        "- Do not use `reports/plots/combined_plot_data.csv` as a canonical source. It is stale and incomplete.",
        "- Mixed batching appears only in exploratory notebooks and should not be presented as part of the final production comparison.",
        "- Chapter 5 generation figures use the two aggregated generation summary CSVs consistently, even though those aggregates may conflict with some per-seed JSON files.",
        "",
        "| Figure | File Path | Chapter | Suggested Placement | Caption | Source Data / File | Caveat |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record.number} | `reports/thesis_figures/{record.filename}` | {record.chapter} | {record.placement} | {record.caption} | {record.source_data} | {record.caveat or 'None'} |"
        )
    FIGURE_INDEX_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme_note() -> None:
    lines = [
        "# Thesis Figures",
        "",
        "This directory contains the regenerated thesis-grade figure set.",
        "",
        "## Notes",
        "",
        "- Do not use old plots in `reports/plots/` directly in the thesis.",
        "- `reports/plots/combined_plot_data.csv` is stale and incomplete and should not be used as a canonical source.",
        "- Mixed batching appears only in exploratory notebooks and should not be presented as part of the final production comparison.",
        "- Chapter 5 generation figures use the two aggregated generation summary CSVs consistently, even though those aggregates may conflict with some per-seed JSON files.",
        "",
        "See `FIGURE_INDEX.md` for captions, placement recommendations, sources, and caveats.",
    ]
    README_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    ensure_output_dir()
    master_rows, _ = load_master_summary()
    per_seed_rows = load_per_seed_results()
    generation_rows = load_generation_summary()
    records = [
        create_figure_1_1(),
        create_figure_1_2(),
        create_figure_2_1(),
        create_figure_3_1(),
        create_figure_3_2(),
        create_figure_3_3(),
        create_figure_3_4(),
        create_figure_4_1(),
        create_figure_4_2(),
        create_figure_4_3(),
        create_figure_4_4(),
        create_figure_5_1(master_rows),
        create_figure_5_2(master_rows),
        create_figure_5_3(master_rows),
        create_figure_5_4(master_rows),
        create_figure_5_5(master_rows),
        create_figure_5_6(generation_rows),
        create_figure_5_7(master_rows, per_seed_rows),
    ]
    write_figure_index(records)
    write_readme_note()
    for record in records:
        verify_png(THESIS_FIGURES_DIR / record.filename)
    if not FIGURE_INDEX_MD.exists():
        raise FileNotFoundError(f"FIGURE_INDEX.md was not created: {FIGURE_INDEX_MD}")
    if not README_MD.exists():
        raise FileNotFoundError(f"README.md was not created: {README_MD}")
    print("Created thesis figures:")
    for record in records:
        print(record.filename)
    print(FIGURE_INDEX_MD.name)
    print(README_MD.name)


if __name__ == "__main__":
    main()
