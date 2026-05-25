from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
THESIS_FIGURES_DIR = REPORTS_DIR / "thesis_figures_polished"
MASTER_SUMMARY_CSV = REPORTS_DIR / "master" / "master_summary_table.csv"
PER_SEED_CSV = REPORTS_DIR / "master" / "master_per_seed_results.csv"
GEN_SUMMARY_A_CSV = REPORTS_DIR / "generation_eval_summaries" / "generation_metrics_summary_table.csv"
GEN_SUMMARY_B_CSV = REPORTS_DIR / "generation_eval_5k" / "generation_metrics_summary_table.csv"
FIGURE_INDEX_MD = THESIS_FIGURES_DIR / "FIGURE_INDEX_POLISHED.md"


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
    "Easy->Hard Length": "Easy -> Hard Length",
    "Hard->Easy Length": "Hard -> Easy Length",
    "Hard->Easy Length Longer Training": "Hard -> Easy Length\nLonger Training",
}

METHOD_ORDER_BY_BLOCK = {
    "1k_short": ["Random", "Grouped", "Grouped->Random", "Random->Grouped"],
    "1k_long": ["Random", "Grouped", "Grouped->Random", "Random->Grouped"],
    "3k_long": ["Random", "Grouped", "Grouped->Random", "Random->Grouped"],
    "5k_long": ["Random", "Grouped", "Grouped->Random", "Random->Grouped"],
    "5k_curriculum_length": ["Easy->Hard Length", "Hard->Easy Length"],
    "5k_curriculum_length_longer": ["Hard->Easy Length", "Hard->Easy Length Longer Training"],
}

METHOD_STYLES = {
    "Random": {"facecolor": "#4C78A8", "marker": "o"},
    "Grouped": {"facecolor": "#F28E2B", "marker": "s"},
    "Grouped->Random": {"facecolor": "#59A14F", "marker": "^"},
    "Random->Grouped": {"facecolor": "#E15759", "marker": "D"},
    "Easy->Hard Length": {"facecolor": "#B07AA1", "marker": "P"},
    "Hard->Easy Length": {"facecolor": "#9C755F", "marker": "X"},
    "Hard->Easy Length Longer Training": {"facecolor": "#3B3B3B", "marker": "*"},
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
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.edgecolor": "#7A7A7A",
            "axes.linewidth": 0.9,
            "figure.facecolor": "#FCFCFD",
            "axes.facecolor": "#FCFCFD",
            "savefig.facecolor": "#FCFCFD",
            "savefig.edgecolor": "#FCFCFD",
            "grid.color": "#D9DCE3",
            "grid.linewidth": 0.8,
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
            method = "Hard->Easy Length Longer Training"
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
            if row["Block"] == "5k_curriculum_length_longer" and row["Method"] == "Hard->Easy Length Longer Training"
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


def add_background_panel(ax, x: float, y: float, width: float, height: float, *, facecolor: str = "#F3F5F8") -> None:
    panel = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        linewidth=0.0,
        edgecolor="none",
        facecolor=facecolor,
        zorder=0,
    )
    ax.add_patch(panel)


def add_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "#FFFFFF",
    edgecolor: str = "#6C6F7A",
    linewidth: float = 1.4,
    fontsize: float = 12,
    weight: str = "normal",
    text_color: str = "#222222",
    align: str = "center",
    rounded: bool = True,
    linestyle: str = "-",
) -> None:
    boxstyle = "round,pad=0.018,rounding_size=0.03" if rounded else "square,pad=0.02"
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
    color: str = "#6C6F7A",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=linewidth,
        color=color,
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


def style_axis(ax, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#D9DCE3", linewidth=0.8)


def save_figure(fig, filename: str) -> Path:
    path = THESIS_FIGURES_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="#FCFCFD")
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


def panel_ylim(values: list[float], std_values: list[float] | None = None, margin_fraction: float = 0.20) -> tuple[float, float]:
    if std_values is None:
        std_values = [0.0 for _ in values]
    lower = min(value - std for value, std in zip(values, std_values))
    upper = max(value + std for value, std in zip(values, std_values))
    margin = max((upper - lower) * margin_fraction, 0.0002)
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
    fig, ax = setup_diagram_axis((14, 4.8))
    add_background_panel(ax, 0.03, 0.16, 0.94, 0.66)
    nodes = [
        ("Pretrained\nLLMs", 0.06, "#E8F1FA"),
        ("Instruction\nFine-Tuning", 0.255, "#EDE7F6"),
        ("Mini-Batch\nScheduling", 0.45, "#EAF5EE"),
        ("Research\nGap", 0.645, "#FCEFE2"),
        ("Research\nQuestion", 0.84, "#F5E8E8"),
    ]
    for idx, (label, x, color) in enumerate(nodes):
        add_box(ax, x, 0.36, 0.13, 0.24, label, facecolor=color, weight="semibold")
        if idx < len(nodes) - 1:
            add_arrow(ax, (x + 0.13, 0.48), (nodes[idx + 1][1], 0.48))
    ax.text(
        0.84,
        0.24,
        "Does batch composition or order\naffect instruction fine-tuning?",
        ha="center",
        va="center",
        fontsize=11,
        color="#404040",
    )
    path = save_figure(fig, "figure_1_1_research_motivation_flow.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 1.1",
        filename=path.name,
        chapter="Chapter 1",
        placement="Section 1.1, after the motivation paragraph",
        caption="Research motivation showing how the thesis narrows from instruction fine-tuning to the question of mini-batch scheduling.",
        source_data="Schematic based on the thesis setup.",
        caveat="Conceptual figure only; not derived from result data.",
    )


def create_figure_1_2() -> FigureRecord:
    fig, ax = setup_diagram_axis((12.5, 7.2))
    add_background_panel(ax, 0.05, 0.08, 0.40, 0.82, facecolor="#EEF5FB")
    add_background_panel(ax, 0.55, 0.08, 0.40, 0.82, facecolor="#FBF1EE")
    ax.text(0.25, 0.84, "Included", ha="center", va="center", fontsize=18, weight="bold", color="#2F4B6C")
    ax.text(0.75, 0.84, "Excluded", ha="center", va="center", fontsize=18, weight="bold", color="#7A4A35")
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
    ax.text(0.10, 0.74, "\n".join(f"• {item}" for item in included), ha="left", va="top", fontsize=12.5, linespacing=1.7)
    ax.text(0.60, 0.74, "\n".join(f"• {item}" for item in excluded), ha="left", va="top", fontsize=12.5, linespacing=1.7)
    path = save_figure(fig, "figure_1_2_research_scope_overview.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 1.2",
        filename=path.name,
        chapter="Chapter 1",
        placement="Section 1.5, after the scope definition",
        caption="Scope of the thesis, separating the investigated components from areas outside the study.",
        source_data="Schematic based on the thesis scope.",
        caveat="Conceptual figure only; not derived from result data.",
    )


def create_figure_2_1() -> FigureRecord:
    fig, ax = setup_diagram_axis((13.5, 8.0))
    add_background_panel(ax, 0.05, 0.24, 0.90, 0.66)
    ax.text(0.50, 0.88, "Position in Existing Work", ha="center", va="center", fontsize=18, weight="bold")
    topic_boxes = [
        ("Instruction\nfine-tuning", 0.10, 0.64, "#E8F1FA"),
        ("PEFT /\nLoRA", 0.39, 0.64, "#EFE9F7"),
        ("Data\ncuration", 0.68, 0.64, "#FCEFE2"),
        ("Curriculum\nlearning", 0.10, 0.40, "#EAF5EE"),
        ("Semantic embeddings\n/ retrieval", 0.39, 0.40, "#EEF3DD"),
        ("Evaluation\nmetrics", 0.68, 0.40, "#F7ECEF"),
    ]
    for text, x, y, color in topic_boxes:
        add_box(ax, x, y, 0.20, 0.15, text, facecolor=color, weight="semibold", fontsize=12)
    add_box(
        ax,
        0.22,
        0.05,
        0.56,
        0.14,
        "Thesis gap:\nMini-batch construction and scheduling\nduring instruction fine-tuning",
        facecolor="#2F4B6C",
        edgecolor="#2F4B6C",
        text_color="white",
        weight="semibold",
    )
    add_arrow(ax, (0.50, 0.40), (0.50, 0.19))
    path = save_figure(fig, "figure_2_1_position_in_existing_work.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 2.1",
        filename=path.name,
        chapter="Chapter 2",
        placement="Section 2.x, after the literature positioning discussion",
        caption="Positioning of the study relative to existing work in instruction fine-tuning, PEFT, curriculum learning, semantic retrieval, and evaluation.",
        source_data="Schematic based on the literature-review framing.",
        caveat="Conceptual figure only; not derived from result data.",
    )


def create_figure_3_1() -> FigureRecord:
    fig, ax = setup_diagram_axis((15, 4.5))
    add_background_panel(ax, 0.03, 0.18, 0.94, 0.60)
    steps = [
        ("Dolly\nDataset", 0.05, "#E8F1FA"),
        ("Preprocessing", 0.22, "#EEF3DD"),
        ("Strategy\nConstruction", 0.39, "#EAF5EE"),
        ("LoRA\nFine-Tuning", 0.56, "#EFE9F7"),
        ("Multi-Seed\nEvaluation", 0.73, "#FCEFE2"),
        ("Result\nAnalysis", 0.88, "#F7ECEF"),
    ]
    for idx, (label, x, color) in enumerate(steps):
        width = 0.12 if idx < len(steps) - 1 else 0.08
        add_box(ax, x, 0.34, width, 0.22, label, facecolor=color, weight="semibold", fontsize=11.5)
        if idx < len(steps) - 1:
            add_arrow(ax, (x + width, 0.45), (steps[idx + 1][1], 0.45))
    path = save_figure(fig, "figure_3_1_methodology_workflow.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 3.1",
        filename=path.name,
        chapter="Chapter 3",
        placement="Section 3.1, methodology overview",
        caption="Methodology workflow from dataset preparation through strategy construction, LoRA fine-tuning, multi-seed evaluation, and result analysis.",
        source_data="Schematic based on `configs/experiments/*.yaml`, `src/batching/*.py`, and `src/training/*.py`.",
        caveat="Conceptual figure only; not derived from result data.",
    )


def create_figure_3_2() -> FigureRecord:
    fig, ax = setup_diagram_axis((13.5, 7.2))
    add_background_panel(ax, 0.04, 0.10, 0.26, 0.80, facecolor="#EEF5FB")
    add_background_panel(ax, 0.37, 0.10, 0.26, 0.80, facecolor="#EEF6F0")
    add_background_panel(ax, 0.70, 0.10, 0.26, 0.80, facecolor="#F4EEF8")
    families = [
        ("Static", ["Random", "Grouped"], 0.17, "#2F4B6C"),
        ("Two-phase", ["Grouped -> Random", "Random -> Grouped"], 0.50, "#3E6A4E"),
        ("Length-based", ["Easy -> Hard", "Hard -> Easy"], 0.83, "#6E507C"),
    ]
    for title, items, x, title_color in families:
        ax.text(x, 0.82, title, ha="center", va="center", fontsize=17, weight="bold", color=title_color)
        ax.text(x - 0.09, 0.68, "\n".join(f"• {item}" for item in items), ha="left", va="top", fontsize=13, linespacing=1.8)
    path = save_figure(fig, "figure_3_2_batch_scheduling_strategies.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 3.2",
        filename=path.name,
        chapter="Chapter 3",
        placement="Section 3.2, after the strategy-family description",
        caption="Final strategy families evaluated in the thesis: static batching, two-phase curricula, and length-based curricula.",
        source_data="Schematic based on the final production strategy set.",
        caveat="Mixed batching appears only in exploratory notebooks and is intentionally omitted here.",
    )


def create_figure_3_3() -> FigureRecord:
    fig, ax = setup_diagram_axis((16, 4.8))
    add_background_panel(ax, 0.03, 0.18, 0.94, 0.58)
    labels = [
        ("Instruction\nText", "#E8F1FA"),
        ("Sentence\nEmbeddings", "#EEF3DD"),
        ("Normalization", "#EAF5EE"),
        ("FAISS\nIndexFlatIP", "#EFE9F7"),
        ("Nearest\nNeighbors", "#FCEFE2"),
        ("Grouped\nMini-Batches", "#F7ECEF"),
    ]
    x_positions = [0.05, 0.20, 0.35, 0.50, 0.66, 0.82]
    widths = [0.11, 0.11, 0.11, 0.12, 0.12, 0.11]
    for idx, ((label, color), x, width) in enumerate(zip(labels, x_positions, widths)):
        add_box(ax, x, 0.34, width, 0.20, label, facecolor=color, weight="semibold", fontsize=11)
        if idx < len(labels) - 1:
            add_arrow(ax, (x + width, 0.44), (x_positions[idx + 1], 0.44))
    ax.text(0.50, 0.12, "top_k = 32 for 1k; top_k = 8 for 3k/5k; batch size = 8", ha="center", va="center", fontsize=11, color="#4D4D4D")
    path = save_figure(fig, "figure_3_3_semantic_minibatch_construction.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 3.3",
        filename=path.name,
        chapter="Chapter 3",
        placement="Section 3.3, semantic grouping subsection",
        caption="Semantic mini-batch construction using sentence embeddings, normalized similarity search, and nearest-neighbor grouping.",
        source_data="Schematic based on `src/batching/grouping.py`, `src/batching/index_loader.py`, and the indexing scripts.",
        caveat="The 1k semantic index differs from 3k/5k in effective `top_k`.",
    )


def create_figure_3_4() -> FigureRecord:
    fig, ax = setup_diagram_axis((13.5, 7.2))
    cards = [
        ("Random -> Grouped", "exploration\nthen refinement", 0.07, 0.56, "#E8F1FA"),
        ("Grouped -> Random", "structure\nthen randomization", 0.54, 0.56, "#EAF5EE"),
        ("Easy -> Hard", "shorter / easier\nexamples first", 0.07, 0.18, "#EFE9F7"),
        ("Hard -> Easy", "longer / harder\nexamples first", 0.54, 0.18, "#FCEFE2"),
    ]
    for title, subtitle, x, y, color in cards:
        add_box(ax, x, y, 0.36, 0.20, f"{title}\n{subtitle}", facecolor=color, weight="semibold", fontsize=12)
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
    fig, ax = setup_diagram_axis((16, 4.8))
    add_background_panel(ax, 0.03, 0.18, 0.94, 0.60)
    steps = [
        ("Dolly\nSubsets", 0.04, "#E8F1FA", 0.10),
        ("Tokenization", 0.16, "#EEF3DD", 0.10),
        ("Processed\nDataset", 0.28, "#EAF5EE", 0.11),
        ("Optional\nSemantic Index", 0.41, "#EFE9F7", 0.12),
        ("Sampler", 0.56, "#FCEFE2", 0.09),
        ("Two-Phase\nLoRA Training", 0.68, "#F7ECEF", 0.12),
        ("Aggregation", 0.83, "#EEF5FB", 0.10),
    ]
    for idx, (label, x, color, width) in enumerate(steps):
        add_box(ax, x, 0.34, width, 0.20, label, facecolor=color, weight="semibold", fontsize=11)
        if idx < len(steps) - 1:
            add_arrow(ax, (x + width, 0.44), (steps[idx + 1][1], 0.44))
    add_box(ax, 0.83, 0.08, 0.10, 0.14, "Generation\nEvaluation", facecolor="#F1F1F1", fontsize=10.5, weight="semibold")
    add_arrow(ax, (0.88, 0.34), (0.88, 0.22))
    path = save_figure(fig, "figure_4_1_experimental_pipeline.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 4.1",
        filename=path.name,
        chapter="Chapter 4",
        placement="Section 4.1, experimental setup overview",
        caption="Implementation pipeline from Dolly subsets through tokenization, sampling, training, aggregation, and generation evaluation.",
        source_data="Schematic based on `manifests/project_manifest.json`, `scripts/training/*.py`, `scripts/reporting/aggregate_results.py`, and `scripts/evaluation/evaluate_generation_quality.py`.",
        caveat="Conceptual implementation diagram; not derived from endpoint result values.",
    )


def create_figure_4_2() -> FigureRecord:
    fig, ax = setup_diagram_axis((16, 4.8))
    add_background_panel(ax, 0.03, 0.18, 0.94, 0.60)
    labels = [
        ("Raw\nRows", 0.05, "#E8F1FA", 0.09),
        ("Embedding\nText", 0.17, "#EEF3DD", 0.10),
        ("FAISS Neighbor\nGraph", 0.30, "#EAF5EE", 0.12),
        ("Raw-to-Processed\nAlignment", 0.46, "#EFE9F7", 0.13),
        ("GroupedBatch\nSampler", 0.63, "#FCEFE2", 0.11),
        ("Seen Set\nHandling", 0.77, "#F7ECEF", 0.10),
    ]
    for idx, (label, x, color, width) in enumerate(labels):
        add_box(ax, x, 0.34, width, 0.20, label, facecolor=color, weight="semibold", fontsize=10.8)
        if idx < len(labels) - 1:
            add_arrow(ax, (x + width, 0.44), (labels[idx + 1][1], 0.44))
    add_box(ax, 0.83, 0.08, 0.10, 0.14, "Final\nBatches", facecolor="#F1F1F1", fontsize=10.5, weight="semibold")
    add_arrow(ax, (0.82, 0.34), (0.88, 0.22), connectionstyle="arc3,rad=-0.18")
    path = save_figure(fig, "figure_4_2_semantic_grouping_pipeline.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 4.2",
        filename=path.name,
        chapter="Chapter 4",
        placement="Section 4.6.2, semantic grouping implementation subsection",
        caption="Implementation-specific semantic grouping from raw rows through FAISS retrieval, alignment, sampler logic, and final grouped batches.",
        source_data="Schematic based on `src/batching/grouping.py`, `src/batching/index_loader.py`, `src/data/alignment.py`, and the indexing scripts.",
        caveat="The 1k semantic index uses older metadata and a different effective neighbor width.",
    )


def create_figure_4_3() -> FigureRecord:
    fig, ax = setup_diagram_axis((13, 6.2))
    add_background_panel(ax, 0.07, 0.52, 0.36, 0.30, facecolor="#EEF5FB")
    add_background_panel(ax, 0.57, 0.52, 0.36, 0.30, facecolor="#F7ECEF")
    add_box(ax, 0.10, 0.58, 0.30, 0.18, "Phase 1\nsampler mode + LR 7e-05\nLoRA training", facecolor="#E8F1FA", weight="semibold", fontsize=12)
    add_box(ax, 0.60, 0.58, 0.30, 0.18, "Phase 2\nsampler mode + LR 5e-05\ncontinued LoRA training", facecolor="#F5E8E8", weight="semibold", fontsize=12)
    add_box(ax, 0.42, 0.34, 0.16, 0.12, "Adapter\nhandoff", facecolor="#F3F3F4", weight="semibold", fontsize=11)
    add_box(ax, 0.42, 0.10, 0.18, 0.12, "Final\ncheckpoint", facecolor="#ECECEC", weight="semibold", fontsize=11)
    add_arrow(ax, (0.40, 0.58), (0.50, 0.46))
    add_arrow(ax, (0.60, 0.58), (0.50, 0.46))
    add_arrow(ax, (0.50, 0.34), (0.50, 0.22))
    path = save_figure(fig, "figure_4_3_two_phase_training_flow.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 4.3",
        filename=path.name,
        chapter="Chapter 4",
        placement="Section 4.4, training procedure subsection",
        caption="Two-phase training flow with phase-specific sampler modes, learning rates, adapter handoff, and final checkpoint output.",
        source_data="Schematic based on `src/training/experiment_runner.py`, `src/training/phase_runner.py`, and `src/training/trainer_factory.py`.",
        caveat="Conceptual implementation diagram; not derived from endpoint result values.",
    )


def create_figure_4_4() -> FigureRecord:
    fig, ax = setup_diagram_axis((13.5, 7.2))
    add_box(ax, 0.35, 0.82, 0.30, 0.10, "Production Batch Scheduling Taxonomy", facecolor="#2F4B6C", edgecolor="#2F4B6C", text_color="white", weight="semibold", fontsize=13)
    families = [
        ("Static", ["Random", "Grouped"], 0.08, "#EEF5FB"),
        ("Two-phase", ["Grouped -> Random", "Random -> Grouped"], 0.37, "#EEF6F0"),
        ("Length-based", ["Easy -> Hard Length", "Hard -> Easy Length"], 0.69, "#F5EFF8"),
    ]
    for title, items, x, color in families:
        add_box(ax, x, 0.30, 0.23, 0.38, "", facecolor=color, fontsize=1)
        ax.text(x + 0.115, 0.60, title, ha="center", va="center", fontsize=15, weight="bold")
        ax.text(x + 0.03, 0.50, "\n".join(f"• {item}" for item in items), ha="left", va="top", fontsize=12.5, linespacing=1.7)
    ax.text(0.50, 0.13, "Mixed batching was exploratory only and is not part of the final production comparison.", ha="center", va="center", fontsize=11, color="#444444")
    path = save_figure(fig, "figure_4_4_batch_scheduling_taxonomy.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 4.4",
        filename=path.name,
        chapter="Chapter 4",
        placement="Section 4.6, strategy summary subsection",
        caption="Taxonomy of the production-implemented batch scheduling strategies used in the final experiment matrix.",
        source_data="Schematic based on the final production strategy set and `src/batching/*.py`.",
        caveat="Mixed batching is exploratory only and is not part of the final production comparison.",
    )


def create_figure_5_1(master_rows: list[dict[str, object]]) -> FigureRecord:
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, block in zip(axes.flatten(), BLOCK_ORDER):
        rows = get_rows_for_block(master_rows, block)
        values = [float(row["Mean Eval Loss"]) for row in rows]
        stds = [float(row["Std Eval Loss"]) for row in rows]
        labels = [DISPLAY_METHODS[row["Method"]] for row in rows]
        y_positions = list(range(len(rows)))
        ax.errorbar(
            values,
            y_positions,
            xerr=stds,
            fmt="none",
            ecolor="#7A7A7A",
            elinewidth=1.1,
            capsize=3,
            zorder=1,
        )
        for y_pos, row in zip(y_positions, rows):
            style = METHOD_STYLES[row["Method"]]
            ax.scatter(
                row["Mean Eval Loss"],
                y_pos,
                s=95,
                marker=style["marker"],
                color=style["facecolor"],
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
        ax.set_yticks(y_positions, labels)
        ax.invert_yaxis()
        ax.set_xlim(*panel_ylim(values, stds, margin_fraction=0.22))
        ax.set_title(BLOCK_LABELS[block])
        disable_axis_offset(ax, "x")
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        style_axis(ax, grid_axis="x")
        ax.set_xlabel("Mean eval loss")
    fig.suptitle("Figure 5.1  Mean Evaluation Loss by Method Across Blocks", fontsize=19, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = save_figure(fig, "figure_5_1_mean_eval_loss_by_block.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 5.1",
        filename=path.name,
        chapter="Chapter 5",
        placement="Section 5.2, after the optimization summary tables",
        caption="Mean final evaluation loss across scheduling strategies and experiment blocks. Lower values indicate better final evaluation performance.",
        source_data=str(MASTER_SUMMARY_CSV),
        caveat="Small visual differences should not be overinterpreted.",
    )


def create_figure_5_2(master_rows: list[dict[str, object]]) -> FigureRecord:
    comparable_blocks = ["1k_short", "1k_long", "3k_long", "5k_long"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, block in zip(axes.flatten(), comparable_blocks):
        rows = get_rows_for_block(master_rows, block)
        random_value = next(float(row["Mean Eval Loss"]) for row in rows if row["Method"] == "Random")
        compare_rows = [row for row in rows if row["Method"] != "Random"]
        y_positions = list(range(len(compare_rows)))
        diffs = [float(row["Mean Eval Loss"]) - random_value for row in compare_rows]
        labels = [DISPLAY_METHODS[row["Method"]] for row in compare_rows]
        colors = [METHOD_STYLES[row["Method"]]["facecolor"] for row in compare_rows]
        bars = ax.barh(y_positions, diffs, color=colors, edgecolor="white", linewidth=0.8)
        ax.axvline(0.0, color="#2E2E2E", linewidth=1.5)
        spread = max(max(abs(value) for value in diffs), 0.00015)
        ax.set_xlim(-spread * 1.4, spread * 1.4)
        ax.set_yticks(y_positions, labels)
        ax.invert_yaxis()
        ax.set_title(BLOCK_LABELS[block])
        disable_axis_offset(ax, "x")
        style_axis(ax, grid_axis="x")
        annotate_horizontal_values(ax, diffs, y_positions, [format_plot_value(value, 5) for value in diffs])
        ax.set_xlabel("Eval loss minus Random")
    fig.suptitle("Figure 5.2  Difference from Random Baseline", fontsize=19, y=0.98)
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
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    for ax, block in zip(axes.flatten(), comparable_blocks):
        rows = get_rows_for_block(master_rows, block)
        grouped_to_random = next(row for row in rows if row["Method"] == "Grouped->Random")
        random_to_grouped = next(row for row in rows if row["Method"] == "Random->Grouped")
        g2r_value = float(grouped_to_random["Mean Eval Loss"])
        r2g_value = float(random_to_grouped["Mean Eval Loss"])
        ax.plot([g2r_value, r2g_value], [0.5, 0.5], color="#B5BAC3", linewidth=2.0, zorder=1)
        ax.scatter(g2r_value, 0.5, s=130, marker=METHOD_STYLES["Grouped->Random"]["marker"], color=METHOD_STYLES["Grouped->Random"]["facecolor"], edgecolors="white", linewidths=0.9, zorder=3)
        ax.scatter(r2g_value, 0.5, s=130, marker=METHOD_STYLES["Random->Grouped"]["marker"], color=METHOD_STYLES["Random->Grouped"]["facecolor"], edgecolors="white", linewidths=0.9, zorder=3)
        ax.set_title(BLOCK_LABELS[block])
        ax.set_xlim(*panel_ylim([g2r_value, r2g_value], margin_fraction=0.40))
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        disable_axis_offset(ax, "x")
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        style_axis(ax, grid_axis="x")
        ax.set_xlabel("Mean eval loss")
    legend_handles = [
        plt.Line2D([0], [0], marker=METHOD_STYLES["Grouped->Random"]["marker"], color="w", markerfacecolor=METHOD_STYLES["Grouped->Random"]["facecolor"], markeredgecolor="white", markersize=10, label="Grouped -> Random"),
        plt.Line2D([0], [0], marker=METHOD_STYLES["Random->Grouped"]["marker"], color="w", markerfacecolor=METHOD_STYLES["Random->Grouped"]["facecolor"], markeredgecolor="white", markersize=10, label="Random -> Grouped"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Figure 5.3  Curriculum Ordering Comparison", fontsize=19, y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    path = save_figure(fig, "figure_5_3_curriculum_ordering_comparison.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 5.3",
        filename=path.name,
        chapter="Chapter 5",
        placement="Section 5.4, curriculum-ordering subsection",
        caption="Paired comparison of the two semantic curriculum orderings across the directly comparable experiment blocks.",
        source_data=str(MASTER_SUMMARY_CSV),
        caveat="Differences between the two orderings are small and should be interpreted cautiously.",
    )


def create_figure_5_4(master_rows: list[dict[str, object]]) -> FigureRecord:
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, block in zip(axes.flatten(), BLOCK_ORDER):
        rows = get_rows_for_block(master_rows, block)
        values = [float(row["Mean Phase2 Delta Eval"]) for row in rows]
        labels = [DISPLAY_METHODS[row["Method"]] for row in rows]
        y_positions = list(range(len(rows)))
        ax.axvline(0.0, color="#2E2E2E", linewidth=1.3)
        for y_position, row, value in zip(y_positions, rows, values):
            style = METHOD_STYLES[row["Method"]]
            ax.scatter(value, y_position, s=95, marker=style["marker"], color=style["facecolor"], edgecolors="white", linewidths=0.8, zorder=3)
        spread = max(abs(min(values)), abs(max(values)))
        ax.set_xlim(-spread * 1.35, spread * 0.25 if spread > 0 else 0.01)
        ax.set_yticks(y_positions, labels)
        ax.invert_yaxis()
        ax.set_title(BLOCK_LABELS[block])
        disable_axis_offset(ax, "x")
        style_axis(ax, grid_axis="x")
        ax.set_xlabel("Mean phase-2 delta eval")
    fig.suptitle("Figure 5.4  Phase-2 Delta Eval Comparison", fontsize=19, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = save_figure(fig, "figure_5_4_phase2_delta_eval_comparison.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 5.4",
        filename=path.name,
        chapter="Chapter 5",
        placement="Section 5.4.2, phase-wise behaviour subsection",
        caption="Mean evaluation-loss change during Phase 2 across methods and blocks. More negative values indicate larger Phase-2 improvement.",
        source_data=str(MASTER_SUMMARY_CSV),
        caveat="Block-wise scales differ, so each panel should be read within its own axis range.",
    )


def create_figure_5_5(master_rows: list[dict[str, object]]) -> FigureRecord:
    same_budget_rows = get_rows_for_block(master_rows, "5k_curriculum_length")
    extension_row = get_rows_for_block(master_rows, "5k_curriculum_length_longer")[1]
    rows = same_budget_rows + [extension_row]

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.add_patch(Rectangle((-0.5, 10.88), 2.0, 0.60, facecolor="#F3F6FA", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((2.2, 10.88), 1.2, 0.60, facecolor="#F4F4F4", edgecolor="none", zorder=0))
    x_positions = [0.0, 1.0, 2.8]
    values = [float(row["Mean Eval Loss"]) for row in rows]
    labels = [
        "Easy -> Hard Length\nsame budget",
        "Hard -> Easy Length\nsame budget",
        "Hard -> Easy Length\nlonger training",
    ]

    ax.plot(
        x_positions[:2],
        values[:2],
        color="#AAB1BB",
        linewidth=1.8,
        linestyle="--",
        zorder=1,
    )

    for x_value, row, value in zip(x_positions, rows, values):
        style = METHOD_STYLES[row["Method"]]
        marker_size = 220 if row["Method"] == "Hard->Easy Length Longer Training" else 160
        ax.scatter(
            x_value,
            value,
            s=marker_size,
            marker=style["marker"],
            color=style["facecolor"],
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
        )
        y_offset = 0.022
        ax.text(
            x_value,
            value + y_offset,
            format_plot_value(value, decimals=5),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.axvline(1.9, color="#9AA1AA", linewidth=1.2, linestyle=":")
    ax.text(0.5, 11.435, "Same-budget comparison", ha="center", va="center", fontsize=12, weight="semibold")
    ax.text(2.8, 11.435, "Longer-training extension", ha="center", va="center", fontsize=12, weight="semibold")
    ax.text(
        2.8,
        11.405,
        "exp_032 uses longer training;\nnot same-budget comparable",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#3F3F3F",
    )

    ax.set_xlim(-0.6, 3.4)
    ax.set_ylim(10.88, 11.47)
    ax.set_xticks(x_positions, labels)
    ax.set_ylabel("Mean eval loss")
    ax.set_title("Figure 5.5  Length Curriculum and Extended Training")
    disable_axis_offset(ax, "y")
    style_axis(ax, grid_axis="y")
    fig.tight_layout()
    path = save_figure(fig, "figure_5_5_length_curriculum_extended_training.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 5.5",
        filename=path.name,
        chapter="Chapter 5",
        placement="Section 5.5, length-based curriculum subsection",
        caption="Same-budget comparison of the two length-based curricula, with the longer-training Hard -> Easy extension shown separately.",
        source_data=str(MASTER_SUMMARY_CSV),
        caveat="The `exp_032` point uses a longer training budget and is not same-budget comparable.",
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
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    for ax, (mean_key, std_key, title) in zip(axes.flatten(), metric_specs):
        rows = [row_lookup[method] for method in ordered_methods]
        y_positions = list(range(len(rows)))
        values = [float(row[mean_key]) for row in rows]
        stds = [float(row[std_key]) for row in rows]
        ax.errorbar(values, y_positions, xerr=stds, fmt="none", ecolor="#7A7A7A", elinewidth=1.0, capsize=3, zorder=1)
        for y_position, row, value in zip(y_positions, rows, values):
            style = METHOD_STYLES[row["method"]]
            ax.scatter(value, y_position, s=95, marker=style["marker"], color=style["facecolor"], edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_yticks(y_positions, [DISPLAY_METHODS[row["method"]] for row in rows])
        ax.invert_yaxis()
        style_axis(ax, grid_axis="x")
        metric_min = min(value - std for value, std in zip(values, stds))
        metric_max = max(value + std for value, std in zip(values, stds))
        margin = max((metric_max - metric_min) * 0.25, 0.0002)
        ax.set_xlim(metric_min - margin, metric_max + margin)
        disable_axis_offset(ax, "x")
        ax.set_title(title)
        ax.set_xlabel("Score")
    fig.suptitle("Figure 5.6  Generation Metrics Comparison", fontsize=19, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = save_figure(fig, "figure_5_6_generation_metrics_comparison.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 5.6",
        filename=path.name,
        chapter="Chapter 5",
        placement="Section 5.6, generation quality subsection",
        caption="Generation-quality metrics for 5k batching and length-curriculum experiments, plotted from the aggregated generation summary CSVs.",
        source_data=f"{GEN_SUMMARY_A_CSV}; {GEN_SUMMARY_B_CSV}",
        caveat="Aggregated generation CSVs conflict with some per-seed JSON files; use the aggregated CSVs consistently.",
    )


def create_figure_5_7(master_rows: list[dict[str, object]], per_seed_rows: list[dict[str, object]]) -> FigureRecord:
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
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
            if method == "Hard->Easy Length Longer Training":
                seed_rows = [row for row in per_seed_rows if row["exp_id"] == "exp_032_hard_to_easy_length_longer_training_5k"]
            for seed_row in seed_rows:
                seed = str(seed_row["seed"])
                x_value = x_index + seed_offsets.get(seed, 0.0)
                style_key = method if method in METHOD_STYLES else "Hard->Easy Length"
                style = METHOD_STYLES[style_key]
                ax.scatter(
                    x_value,
                    float(seed_row["final_eval_loss"]),
                    s=48,
                    marker=style["marker"],
                    color=style["facecolor"],
                    edgecolors="white",
                    linewidths=0.8,
                    alpha=0.85,
                    zorder=3,
                )
            mean_row = next(row for row in rows if row["Method"] == method)
            ax.scatter(
                x_index,
                float(mean_row["Mean Eval Loss"]),
                s=110,
                marker="D",
                color="#1F1F1F",
                edgecolors="white",
                linewidths=0.9,
                zorder=4,
            )
        ax.set_xticks(range(len(display_labels)), display_labels, rotation=20, ha="right")
        ax.set_ylim(y_min, y_max)
        disable_axis_offset(ax, "y")
        ax.set_title(BLOCK_LABELS[block])
        style_axis(ax, grid_axis="y")
        ax.set_ylabel("Final eval loss")
    fig.suptitle("Figure 5.7  Seed Variance Plot", fontsize=19, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = save_figure(fig, "figure_5_7_seed_variance_plot.png")
    verify_png(path)
    return FigureRecord(
        number="Figure 5.7",
        filename=path.name,
        chapter="Chapter 5",
        placement="Section 5.7, seed variance and stability subsection",
        caption="Per-seed final evaluation loss across methods and blocks, with block means overlaid. Each experiment includes three seeds.",
        source_data=f"{PER_SEED_CSV}; {MASTER_SUMMARY_CSV}",
        caveat="Only three seeds were used per experiment.",
    )


def write_figure_index(records: list[FigureRecord]) -> None:
    lines = [
        "# Figure Index Polished",
        "",
        "This directory contains the polished thesis-quality figure set generated from code.",
        "",
        "## Notes",
        "",
        "- Chapter 5 plots use canonical CSV files only.",
        f"- Canonical optimization source: `{MASTER_SUMMARY_CSV.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Canonical per-seed source: `{PER_SEED_CSV.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Canonical generation sources: `{GEN_SUMMARY_A_CSV.relative_to(PROJECT_ROOT).as_posix()}` and `{GEN_SUMMARY_B_CSV.relative_to(PROJECT_ROOT).as_posix()}`",
        "- `reports/plots/combined_plot_data.csv` is not used.",
        "- Old PNG plots in `reports/plots/` are not used.",
        "- Mixed batching is exploratory only and is not part of the final production comparison.",
        "",
        "| Figure | File Path | Chapter | Suggested Placement | Caption | Source File | Caveat |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record.number} | `reports/thesis_figures_polished/{record.filename}` | {record.chapter} | {record.placement} | {record.caption} | {record.source_data} | {record.caveat or 'None'} |"
        )
    FIGURE_INDEX_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    for record in records:
        verify_png(THESIS_FIGURES_DIR / record.filename)
    if not FIGURE_INDEX_MD.exists():
        raise FileNotFoundError(f"FIGURE_INDEX_POLISHED.md was not created: {FIGURE_INDEX_MD}")
    print("Created polished thesis figures:")
    for record in records:
        print(record.filename)
    print(FIGURE_INDEX_MD.name)


if __name__ == "__main__":
    main()
