import matplotlib.pyplot as plt
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
sizes_2 = counts_2
sizes_3 = counts_3
sizes_4 = counts_4

fig, ax = plt.subplots(figsize=(6, 6))

# 内圈 2-hop
ax.pie(
    sizes_2,
    radius=0.6,
    colors=colors,
    startangle=90,
    wedgeprops=dict(width=0.2, edgecolor="white"),
)

# 中圈 3-hop
ax.pie(
    sizes_3,
    radius=0.8,
    colors=colors,
    startangle=90,
    wedgeprops=dict(width=0.2, edgecolor="white"),
)

# 外圈 4-hop
ax.pie(
    sizes_4,
    radius=1.0,
    colors=colors,
    startangle=90,
    wedgeprops=dict(width=0.2, edgecolor="white"),
)

ax.text(0, 0, "2 / 3 / 4-hop\n(2000 errors each)",
        ha="center", va="center", fontsize=10)

ax.set(aspect="equal", title="Nested donut of error-type counts")

# 图例
legend_labels = [f"{lab}" for lab in labels]
ax.legend(
    loc="lower center",
    labels=legend_labels,
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, -0.05)
)

plt.tight_layout()
plt.show()
