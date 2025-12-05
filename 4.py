import matplotlib.pyplot as plt
import numpy as np

# 百分比
format_pct = [6.5, 8.9, 8.4]
entity_pct = [21.3, 20.3, 20.0]
facts_pct  = [56.4, 59.1, 65.6]
decomp_pct = [9.8,  7.9,  7.0]

# 换成数量（2000 个错误样本）
def to_counts(pcts, total=2000):
    return [total * p / 100.0 for p in pcts]

counts_2 = to_counts([format_pct[0], entity_pct[0], facts_pct[0], decomp_pct[0]])
counts_3 = to_counts([format_pct[1], entity_pct[1], facts_pct[1], decomp_pct[1]])
counts_4 = to_counts([format_pct[2], entity_pct[2], facts_pct[2], decomp_pct[2]])

labels = ["Format", "Entity", "Facts", "Decomp"]
colors = ["#8da0cb", "#66c2a5", "#fc8d62", "#e78ac3"]  # 和谐配色

fig, axes = plt.subplots(1, 3, figsize=(6, 6))  # 正方形画布
hops = ["2-hop", "3-hop", "4-hop"]
all_counts = [counts_2, counts_3, counts_4]

for ax, hop, counts in zip(axes, hops, all_counts):
    # 画 donut pie
    wedges, texts = ax.pie(
        counts,
        labels=None,        # 外圈不写 label，避免拥挤
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.3, edgecolor="white")
    )
    # 在中心写 hop 名称
    ax.text(0, 0, hop, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")

# 统一图例放在外面（用 labels 显示）
fig.legend(
    wedges,
    labels,
    loc="lower center",
    ncol=4,
    frameon=False
)

fig.suptitle("Error-type counts (2000 wrong cases per hop)", fontsize=12)
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.show()
