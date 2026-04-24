# scripts/collect_research_bundle_local.py

from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CONFIGS_DIR = PROJECT_ROOT / "configs" / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"

EXPORT_ROOT = PROJECT_ROOT / "exports" / "research_bundle_local"


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def collect_run_files() -> None:
    run_summaries_dir = EXPORT_ROOT / "run_summaries"
    run_manifests_dir = EXPORT_ROOT / "run_manifests"
    phase_summaries_dir = EXPORT_ROOT / "phase_summaries"

    experiment_dirs = sorted([p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir()])

    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name

        # only real experiment dirs
        if not exp_name.startswith("exp_"):
            continue

        seed_dirs = sorted([p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])

        for seed_dir in seed_dirs:
            seed_name = seed_dir.name

            run_summary = seed_dir / "run_summary.json"
            if run_summary.exists():
                dst = run_summaries_dir / f"run_summary__{exp_name}__{seed_name}.json"
                safe_copy(run_summary, dst)

            run_manifest = seed_dir / "run_manifest.json"
            if run_manifest.exists():
                dst = run_manifests_dir / f"run_manifest__{exp_name}__{seed_name}.json"
                safe_copy(run_manifest, dst)

            phase_dirs = sorted([p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("phase_")])
            for phase_dir in phase_dirs:
                phase_name = phase_dir.name
                phase_summary = phase_dir / "phase_summary.json"
                if phase_summary.exists():
                    dst = phase_summaries_dir / f"phase_summary__{exp_name}__{seed_name}__{phase_name}.json"
                    safe_copy(phase_summary, dst)


def collect_configs() -> None:
    export_configs_dir = EXPORT_ROOT / "experiment_configs"
    yaml_files = sorted(CONFIGS_DIR.glob("exp_*.yaml"))
    for yaml_file in yaml_files:
        safe_copy(yaml_file, export_configs_dir / yaml_file.name)


def collect_reports() -> None:
    export_reports_dir = EXPORT_ROOT / "reports"
    if REPORTS_DIR.exists():
        for file_path in REPORTS_DIR.iterdir():
            if file_path.is_file():
                safe_copy(file_path, export_reports_dir / file_path.name)


def collect_plots() -> None:
    export_plots_dir = EXPORT_ROOT / "plots"
    if PLOTS_DIR.exists():
        for file_path in PLOTS_DIR.iterdir():
            if file_path.is_file():
                safe_copy(file_path, export_plots_dir / file_path.name)


def write_index_file() -> None:
    index_path = EXPORT_ROOT / "README_bundle.txt"

    lines = []
    lines.append("Research Bundle (Local)")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append("")
    lines.append("Included folders:")
    lines.append("- run_summaries/")
    lines.append("- run_manifests/")
    lines.append("- phase_summaries/")
    lines.append("- experiment_configs/")
    lines.append("- reports/")
    lines.append("- plots/")
    lines.append("")

    def count_files(folder: Path) -> int:
        if not folder.exists():
            return 0
        return len([p for p in folder.iterdir() if p.is_file()])

    for name in [
        "run_summaries",
        "run_manifests",
        "phase_summaries",
        "experiment_configs",
        "reports",
        "plots",
    ]:
        folder = EXPORT_ROOT / name
        lines.append(f"{name}: {count_files(folder)} files")

    index_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if EXPORT_ROOT.exists():
        shutil.rmtree(EXPORT_ROOT)

    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

    collect_run_files()
    collect_configs()
    collect_reports()
    collect_plots()
    write_index_file()

    print(f"Bundle created at: {EXPORT_ROOT}")
    print("Subfolders:")
    for folder in sorted(EXPORT_ROOT.iterdir()):
        if folder.is_dir():
            n_files = len([p for p in folder.iterdir() if p.is_file()])
            print(f"  - {folder.name}: {n_files} files")


if __name__ == "__main__":
    main()