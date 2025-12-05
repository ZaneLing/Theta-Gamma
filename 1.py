import matplotlib.pyplot as plt
import numpy as np

# 百分比数据
format_pct = [6.5, 8.9, 8.4]
entity_pct = [21.3, 20.3, 20.0]
facts_pct  = [56.4, 59.1, 65.6]
decomp_pct = [9.8,  7.9,  7.0]

hops = ["2-hop", "3-hop", "4-hop"]
x = np.arange(len(hops))
width = 0.18  # 每根小柱子的宽度

plt.figure(figsize=(5, 5))  # 正方形画布
ax = plt.gca()

ax.bar(x - 1.5*width, format_pct, width, label="Format",  color="#8da0cb")
ax.bar(x - 0.5*width, entity_pct, width, label="Entity",  color="#66c2a5")
ax.bar(x + 0.5*width, facts_pct,  width, label="Facts",   color="#fc8d62")
ax.bar(x + 1.5*width, decomp_pct, width, label="Decomp",  color="#e78ac3")

ax.set_xticks(x)
ax.set_xticklabels(hops)
ax.set_ylabel("Proportion among wrong cases (%)")
ax.set_ylim(0, 80)  # 根据你数据调一下上限
ax.set_title("Error-type distribution over 2/3/4-hop wrong cases")
ax.legend(frameon=False)

plt.tight_layout()
plt.show()
