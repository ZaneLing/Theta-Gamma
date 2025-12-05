import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 读入数据并重命名列
df = pd.read_excel("result.xlsx")
df = df.rename(columns={
    "Unnamed: 0": "dataset",
    "总样本数目": "total_samples",
    "正确率": "acc",
    "总正确个数": "num_correct",
    "总错误个数": "num_wrong",
    "问题分解错误个数": "num_decomp_err",
    "问题分解错误占比": "ratio_decomp_err",
    "证据错误或者不全个数": "num_evi_err",
    "证据错误占比": "ratio_evi_err",
    "答案对齐和整合错误": "num_align_err",
    "答案对齐错误": "ratio_align_err"
})

# x 轴：2hop / 3hop / 4hop
datasets = df["dataset"].tolist()
hops = [d.split("-")[-1] for d in datasets]  # ["2hop","3hop","4hop"]
# 组中心位置
group_gap = 0.4
x = np.arange(len(hops)) * (1.0 + group_gap)

# 2. 计算三类错误在“所有样本”中的占比（直接用你表中的比例）
error_ratio_decomp = df["ratio_decomp_err"].values * 100  # %
error_ratio_evi    = df["ratio_evi_err"].values * 100
error_ratio_align  = df["ratio_align_err"].values * 100

# 只画柱状图
fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))

# -------- 左图：(a) 三类错误在所有样本中的占比（分组柱状图） --------
# 颜色改为 0-255 的三位整数输入，内部转换到 0-1
def _rgb(rgb_255):
    return tuple(v / 255 for v in rgb_255)

color_decomp = _rgb((186, 204, 217))   
color_align   = _rgb((86, 152, 195))   
color_evi  = _rgb((80, 101, 154))   

width = 0.18  # 柱宽
inner_gap = 0.05  # 组内柱子之间的间隔
ax1.bar(
    x - width - inner_gap,
    error_ratio_decomp,
    width,
    label="Decomposition Error",
    color=color_decomp,
)
ax1.bar(
    x,
    error_ratio_align,
    width,
    label="Integration Error",
    color=color_align,
)
ax1.bar(
    x + width + inner_gap,
    error_ratio_evi,
    width,
    label="Evidence Error",
    color=color_evi,
)

ax1.set_xticks(x)
ax1.set_xticklabels(hops)
ax1.set_ylabel("Proportion of all wrong cases (%)")
ax1.set_title("Error type ratios on MuSiQue")
ax1.set_ylim(0, max(error_ratio_evi) * 1.25)
ax1.legend(frameon=False, fontsize=8)
ax1.grid(axis="y", linestyle="--", alpha=0.3)

fig.tight_layout()
fig.savefig("musique_faithfulness_gap_bar.pdf", bbox_inches="tight")
plt.show()
