import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [6.5,  21.3, 56.4, 9.8],   # 2-hop
    [8.9,  20.3, 59.1, 7.9],   # 3-hop
    [8.4,  20.0, 65.6, 7.0],   # 4-hop
])

row_labels = ["2-hop", "3-hop", "4-hop"]
col_labels = ["Format", "Entity", "Facts", "Decomp"]

plt.figure(figsize=(5, 5))
ax = plt.gca()

im = ax.imshow(data, cmap="YlOrRd")  # 黄色→红色的渐变

# 坐标刻度
ax.set_xticks(np.arange(len(col_labels)))
ax.set_yticks(np.arange(len(row_labels)))
ax.set_xticklabels(col_labels)
ax.set_yticklabels(row_labels)

# 在格子里写数值
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{data[i, j]:.1f}%",
                ha="center", va="center", fontsize=9, color="black")

ax.set_title("Error-type heatmap over wrong cases")
plt.colorbar(im, fraction=0.046, pad=0.04, label="Proportion (%)")

plt.tight_layout()
plt.show()
