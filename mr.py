import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# -----------------------
# 1. 方法和类别定义
# -----------------------
methods = [
    "SP", "FSM", "CoT",
    "Single-step", "Self-Ask", "IRCoT", "FLARE", "ProbTree", "EfficientRAG", "BeamAggR",
    "CoA", "HippoRAG", "GEAR", "PRISM", "RopMura", "KAG", "ReAgent", "Search-o1", "BELLE",
    "TG-HORA"
]

categories = [
    "Prompt", "Prompt", "Prompt",
    "Retrieval", "Retrieval", "Retrieval", "Retrieval", "Retrieval", "Retrieval", "Retrieval",
    "Agent", "Agent", "Agent", "Agent", "Agent", "Agent", "Agent", "Agent", "Agent",
    "TG-HORA"
]

datasets = ["HotpotQA", "2WikiQA", "MuSiQue"]

# -----------------------
# 2. F1 数据（你的表里抄过来的）
# -----------------------
f1_hotpot = [
    38.9, 46.0, 46.5,
    55.3, 49.4, 56.2, 56.1, 60.4, 57.9, 62.9,
    55.8, 71.7, 54.6, 67.0, 53.1, 78.2, 79.5, 57.3, 66.5,
    76.4
]

f1_2wiki = [
    33.9, 49.3, 42.3,
    42.9, 46.9, 56.8, 60.1, 67.9, 51.6, 71.6,
    69.7, 72.5, 52.3, 57.0, 63.2, 78.1, 79.3, 71.4, 75.7,
    84.6
]

f1_musique = [
    26.2, 26.2, 22.5,
    10.6, 8.8, 19.2, 27.1, 30.9, 23.6, 36.8,
    36.1, 50.7, 20.9, 41.8, 24.6, 48.9, 51.5, 28.2, 42.1,
    56.7
]

f1_all = [f1_hotpot, f1_2wiki, f1_musique]

# -----------------------
# 3. EM 数据
# -----------------------
em_hotpot = [
    32.1, 33.1, 40.5,
    48.7, 44.5, 51.2, 50.8, 56.3, 52.9, 55.6,
    39.1, 52.8, 50.4, 54.2, 49.2, 60.3, 63.0, 45.2, 59.2,
    72.1
]

em_2wiki = [
    27.8, 36.1, 36.2,
    38.1, 40.5, 50.7, 58.2, 64.3, 47.7, 66.1,
    57.5, 63.3, 47.4, 48.6, 58.8, 68.1, 71.1, 58.0, 69.7,
    81.6
]

em_musique = [
    26.4, 22.2, 21.2,
    22.1, 24.4, 31.4, 40.9, 41.2, 32.7, 45.9,
    23.9, 35.3, 35.1, 31.2, 41.1, 34.8, 37.1, 16.6, 50.5,
    52.5
]

em_all = [em_hotpot, em_2wiki, em_musique]

# -----------------------
# 4. 按类别上色
# -----------------------
color_map = {
    "Prompt":   "#8da0cb",  # 蓝
    "Retrieval":"#66c2a5",  # 绿
    "Agent":    "#fc8d62",  # 橙
    "TG-HORA":  "#e78ac3"   # 粉
}
colors = [color_map[c] for c in categories]

legend_elems = [
    Patch(facecolor=color_map["Prompt"],    label="Prompt-based"),
    Patch(facecolor=color_map["Retrieval"], label="Retrieval-augmented"),
    Patch(facecolor=color_map["Agent"],     label="Agent-based"),
    Patch(facecolor=color_map["TG-HORA"],   label="TG-HORA")
]

x = np.arange(len(methods))

# -----------------------
# 5. 画 3 张图：每张图一排两个子图（左 EM，右 F1）
# -----------------------
for dataset, em, f1, fname in zip(
    datasets, em_all, f1_all,
    ["hotpot", "2wiki", "musique"]
):
    fig, (ax_em, ax_f1) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # 左：EM
    ax_em.bar(x, em, color=colors)
    ax_em.set_title(f"{dataset} (EM)")
    ax_em.set_xticks(x)
    ax_em.set_xticklabels(methods, rotation=90, fontsize=7)
    ax_em.set_ylabel("EM score")
    ax_em.set_ylim(0, 90)
    ax_em.grid(axis="y", linestyle="--", alpha=0.3)

    # 右：F1
    ax_f1.bar(x, f1, color=colors)
    ax_f1.set_title(f"{dataset} (F1)")
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(methods, rotation=90, fontsize=7)
    ax_f1.set_ylim(0, 90)
    ax_f1.grid(axis="y", linestyle="--", alpha=0.3)

    # 统一图例放在上方中间
    fig.legend(handles=legend_elems, loc="upper center",
               ncol=4, frameon=False, fontsize=9)

    #fig.suptitle(f"Main results on {dataset}", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(f"mr_{fname}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"mr_{fname}.pdf", bbox_inches="tight")

# 不再 plt.show(); figures are saved above
