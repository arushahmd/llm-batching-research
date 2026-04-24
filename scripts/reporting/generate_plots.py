from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
out = PROJECT_ROOT / "reports" / "plots"
out.mkdir(parents=True, exist_ok=True)

# Reconstructed from aggregated summaries shared in the conversation
short_1k = pd.DataFrame([
    {"Method": "Random", "Mean Eval Loss": 9.616399129231771, "Std Eval Loss": 0.00012846433173675242, "Mean Phase2 ΔEval": -0.0013834635416666667},
    {"Method": "Grouped", "Mean Eval Loss": 9.61652692159017, "Std Eval Loss": 0.00021044965469782815, "Mean Phase2 ΔEval": -0.0013230641682942708},
    {"Method": "Grouped→Random", "Mean Eval Loss": 9.616591453552246, "Std Eval Loss": 1.2615925364802315e-05, "Mean Phase2 ΔEval": -0.001311937967936198},
    {"Method": "Random→Grouped", "Mean Eval Loss": 9.616386731465658, "Std Eval Loss": 9.03947098320946e-05, "Mean Phase2 ΔEval": -0.0013634363810221355},
])

long_1k = pd.DataFrame([
    {"Method": "Random", "Mean Eval Loss": 9.611238797505697, "Std Eval Loss": 0.00021307810464100992, "Mean Phase2 ΔEval": -0.0039908091227213545},
    {"Method": "Grouped", "Mean Eval Loss": 9.611552556355795, "Std Eval Loss": 0.00010812468756713155, "Mean Phase2 ΔEval": -0.0038159688313802085},
    {"Method": "Grouped→Random", "Mean Eval Loss": 9.611759503682455, "Std Eval Loss": 0.00021096115141597821, "Mean Phase2 ΔEval": -0.0037781397501627603},
    {"Method": "Random→Grouped", "Mean Eval Loss": 9.611469268798828, "Std Eval Loss": 0.0002447879672933447, "Mean Phase2 ΔEval": -0.003803253173828125},
])

long_3k = pd.DataFrame([
    {"Method": "Random", "Mean Eval Loss": 11.423059145609537, "Std Eval Loss": 0.0022963785111432176, "Mean Phase2 ΔEval": -0.03380934397379557},
    {"Method": "Grouped", "Mean Eval Loss": 11.425052642822266, "Std Eval Loss": 0.0006824924120630219, "Mean Phase2 ΔEval": -0.032903035481770836},
    {"Method": "Grouped→Random", "Mean Eval Loss": 11.424866358439127, "Std Eval Loss": 0.001265824868793103, "Mean Phase2 ΔEval": -0.03327242533365885},
    {"Method": "Random→Grouped", "Mean Eval Loss": 11.425841649373373, "Std Eval Loss": 0.000684287960140967, "Mean Phase2 ΔEval": -0.032347679138183594},
])

# Save combined source data
combined = pd.concat(
    [
        short_1k.assign(Block="1k_short"),
        long_1k.assign(Block="1k_long"),
        long_3k.assign(Block="3k_long"),
    ],
    ignore_index=True,
)
combined.to_csv(out / "combined_plot_data.csv", index=False)

order = ["Random", "Grouped", "Grouped→Random", "Random→Grouped"]

def make_bar_plot(df: pd.DataFrame, title: str, filename: str):
    d = df.set_index("Method").loc[order].reset_index()
    plt.figure(figsize=(8, 5))
    plt.bar(d["Method"], d["Mean Eval Loss"], yerr=d["Std Eval Loss"], capsize=5)
    plt.ylabel("Mean Eval Loss")
    plt.title(title)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out / filename, dpi=200, bbox_inches="tight")
    plt.close()

def make_phase_improvement_plot(df: pd.DataFrame, title: str, filename: str):
    d = df.set_index("Method").loc[order].reset_index()
    plt.figure(figsize=(8, 5))
    plt.bar(d["Method"], d["Mean Phase2 ΔEval"])
    plt.ylabel("Mean Phase 2 ΔEval")
    plt.title(title)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out / filename, dpi=200, bbox_inches="tight")
    plt.close()

make_bar_plot(short_1k, "1k Short Training: Mean Eval Loss by Method", "plot_1k_short_eval_loss.png")
make_bar_plot(long_1k, "1k Long Training: Mean Eval Loss by Method", "plot_1k_long_eval_loss.png")
make_bar_plot(long_3k, "3k Long Training: Mean Eval Loss by Method", "plot_3k_long_eval_loss.png")

make_phase_improvement_plot(short_1k, "1k Short Training: Phase 2 Improvement", "plot_1k_short_phase2_delta.png")
make_phase_improvement_plot(long_1k, "1k Long Training: Phase 2 Improvement", "plot_1k_long_phase2_delta.png")
make_phase_improvement_plot(long_3k, "3k Long Training: Phase 2 Improvement", "plot_3k_long_phase2_delta.png")

# Cross-block comparison line plot by method
plt.figure(figsize=(9, 5))
for method in order:
    d = combined[combined["Method"] == method].set_index("Block").loc[["1k_short", "1k_long", "3k_long"]].reset_index()
    plt.plot(d["Block"], d["Mean Eval Loss"], marker="o", label=method)
plt.ylabel("Mean Eval Loss")
plt.title("Mean Eval Loss Across Experiment Blocks")
plt.legend()
plt.tight_layout()
plt.savefig(out / "plot_cross_block_eval_trends.png", dpi=200, bbox_inches="tight")
plt.close()

print("Created files:")
for p in sorted(out.glob("plot_*.png")):
    print(p.name)
print((out / "combined_plot_data.csv").name)