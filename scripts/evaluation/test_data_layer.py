# scripts/test_data_layer.py

from pathlib import Path

from src.data.alignment import build_alignment, validate_alignment
from src.data.processed_loader import inspect_processed_dataset, load_processed_dataset
from src.data.raw_loader import build_embedding_text, inspect_raw_dataset, load_raw_dataset


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    raw_path = project_root / "data" / "raw" / "dolly_15k"
    processed_path = (
        project_root
        / "data"
        / "processed"
        / "dolly_small_1k__flan-t5-small__tok_with_rawidx_v1"
    )

    raw_dataset = load_raw_dataset(raw_path)
    processed_dataset = load_processed_dataset(processed_path)

    print("RAW INFO:", inspect_raw_dataset(raw_dataset))
    print("PROCESSED INFO:", inspect_processed_dataset(processed_dataset))

    alignment = build_alignment(
        processed_train_ds=processed_dataset["train"],
        raw_train_ds=raw_dataset["train"],
    )
    print("ALIGNMENT:", validate_alignment(alignment))

    example = alignment.get_joint_view(processed_idx=0)
    print("RAW IDX:", example["raw_idx"])
    print("INSTRUCTION:", example["raw"]["instruction"])
    print("EMBEDDING TEXT:", build_embedding_text(example["raw"]))


if __name__ == "__main__":
    main()