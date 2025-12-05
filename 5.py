import numpy as np
import matplotlib.pyplot as plt

# 四个维度
labels = ["Format", "Entity", "Facts", "Decomp"]
num_vars = len(labels)

# 各 hop 百分比
hop2 = [6.5, 21.3, 56.4, 9.8]
hop3 = [8.9, 20.3, 59.1, 7.9]
hop4 = [8.4, 20.0, 65.6, 7.0]

# 角度：0 ~ 2π，首尾要闭合，所以最后再加一个起点
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

def close(values):
    return values + values[:1]

hop2_c = close(hop2)
hop3_c = close(hop3)
hop4_c = close(hop4)

plt.figure(figsize=(5, 5))
ax = plt.subplot(111, polar=True)

# 画网格
ax.set_theta_offset(np.pi / 2)      # 让第一个维度在正上方
ax.set_theta_direction(-1)          # 顺时针
ax.set_thetagrids(np.degrees(angles[:-1]), labels)

# 半径范围
ax.set_ylim(0, 70)                  # 你的数据最高是 65.6，可以设 70 或 80

# 三条曲线（颜色选了一套比较和谐的）
ax.plot(angles, hop2_c, label="2-hop", color="#66c2a5", linewidth=2)
ax.fill(angles, hop2_c, alpha=0.15, color="#66c2a5")

ax.plot(angles, hop3_c, label="3-hop", color="#8da0cb", linewidth=2)
ax.fill(angles, hop3_c, alpha=0.15, color="#8da0cb")

ax.plot(angles, hop4_c, label="4-hop", color="#fc8d62", linewidth=2)
ax.fill(angles, hop4_c, alpha=0.15, color="#fc8d62")

# 其他美化
ax.set_title("Error-type distribution over hop lengths", fontsize=12, pad=20)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.10), frameon=False)

plt.tight_layout()
plt.show()
