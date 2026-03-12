import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  # 加这行
import seaborn as sns
import os

# ========== 设置输出文件夹 ==========
output_dir = "strategy_comparison_figures"
os.makedirs(output_dir, exist_ok=True)

# ========== 设置Nature风格 ==========
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 4
plt.rcParams["ytick.major.size"] = 4

# 字体加粗设置
label_font = {"fontsize": 12, "fontweight": "bold"}
tick_font = {"fontsize": 11, "fontweight": "bold"}
title_font = {"fontsize": 13, "fontweight": "bold"}
legend_font = FontProperties(family="Arial", size=10, weight="bold")

# ========== 加载数据 ==========
df = pd.read_csv("all_strategy_comparison.csv")

# ========== 设定颜色 ==========
palette = {
    "KFold": "#65a30d",         # 绿色
    "ShuffleSplit": "#f97316",  # 橙色
    "LeaveOneElement": "#3b82f6"  # 蓝色
}

# ========== 绘图函数 ==========
def plot_metric(metric, ylabel, filename):
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=df,
        x="Model",
        y=metric,
        hue="Strategy",
        palette=palette,
        edgecolor='black'
    )

    plt.xlabel("Model", **label_font)
    plt.ylabel(ylabel, **label_font)
    plt.xticks(**tick_font)
    plt.yticks(**tick_font)
    plt.title(f"Comparison of {ylabel} Across Strategies", **title_font)
    plt.legend(title="Validation Strategy", prop=legend_font, title_fontsize=11)
    plt.savefig(os.path.join(output_dir, filename), dpi=600)
    plt.close()

# ========== 分别绘制 MAE / RMSE / R² ==========
plot_metric("MAE", "MAE", "mae_comparison.png")
plot_metric("RMSE", "RMSE", "rmse_comparison.png")
plot_metric("R2", "R² Score", "r2_comparison.png")