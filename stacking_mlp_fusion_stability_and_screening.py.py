import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# === 设置 Nature 风格 ===
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['figure.dpi'] = 150

# === 读取数据 ===
df = pd.read_csv("2.csv")

# === 特征构造 ===
df["EN_div_AR"] = df["Electronegativity"] / df["AtomicRadius"]
df["IE_div_AR"] = df["FirstIonizationEnergy"] / df["AtomicRadius"]
df["Group_x_Period"] = df["Group"] * df["period"]
df["EN_x_IE"] = df["Electronegativity"] * df["FirstIonizationEnergy"]
df["ReplacedAtomType"] = df["SourceSite"].apply(lambda x: 0 if "Site_1" in x or "Site_2" in x else 1)
df["ReplacedAtomType_x_EN"] = df["ReplacedAtomType"] * df["Electronegativity"]
df["ReplacedAtomType_x_AR"] = df["ReplacedAtomType"] * df["AtomicRadius"]

# === 特征与目标 ===
feature_cols = [
    "Group", "Electronegativity", "AtomicRadius", "FirstIonizationEnergy",
    "SiteTypeCode", "period", "block",
    "EN_div_AR", "IE_div_AR", "Group_x_Period", "EN_x_IE",
    "ReplacedAtomType", "ReplacedAtomType_x_EN", "ReplacedAtomType_x_AR"
]
X = df[feature_cols].values
y = df["Energy (eV)"].values

# === 数据归一化 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 模型定义（你已验证过的最优参数）===
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    learning_rate_init=0.0006,
    max_iter=8000,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)

# === 基学习器（保持一致） ===
rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=3, max_features='sqrt', random_state=42)
gbdt = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
svr = SVR(C=10, epsilon=0.2, gamma='scale')
base_learners = [("rf", rf), ("gbdt", gbdt), ("svr", svr)]

# === 重复训练并记录结果 ===
n_runs = 50
all_scores = []
good_scores = []

for seed in range(n_runs):
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=seed)

    stack = StackingRegressor(
        estimators=base_learners,
        final_estimator=mlp,
        passthrough=False,
        n_jobs=-1
    )
    stack.fit(X_train, y_train)
    y_pred = stack.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    all_scores.append({"Seed": seed, "R2": r2, "MAE": mae})

    if r2 > 0.75:
        good_scores.append({"Seed": seed, "R2": r2, "MAE": mae})

# === 保存结果 ===
df_all = pd.DataFrame(all_scores)
df_good = pd.DataFrame(good_scores)
df_all.to_csv("mlp_fusion_all_50.csv", index=False)
df_good.to_csv("mlp_fusion_good.csv", index=False)

# === 加载你上传的全元素表 ===
df_screen = pd.read_csv("all_elements.csv")

# === 按照训练集相同方式构造特征 ===
df_screen["EN_div_AR"] = df_screen["Electronegativity"] / df_screen["AtomicRadius"]
df_screen["IE_div_AR"] = df_screen["FirstIonizationEnergy"] / df_screen["AtomicRadius"]
df_screen["Group_x_Period"] = df_screen["Group"] * df_screen["period"]
df_screen["EN_x_IE"] = df_screen["Electronegativity"] * df_screen["FirstIonizationEnergy"]
df_screen["ReplacedAtomType"] = df_screen["SourceSite"].apply(lambda x: 0 if "Site_1" in x or "Site_2" in x else 1)
df_screen["ReplacedAtomType_x_EN"] = df_screen["ReplacedAtomType"] * df_screen["Electronegativity"]
df_screen["ReplacedAtomType_x_AR"] = df_screen["ReplacedAtomType"] * df_screen["AtomicRadius"]

# === 使用和训练集相同的特征列 ===
X_screen = df_screen[feature_cols].values
X_screen_scaled = scaler.transform(X_screen)

# === 用训练好的 stack 模型进行预测 ===
y_pred = stack.predict(X_screen_scaled)

# === 保存预测结果 ===
df_screen["Predicted_Energy"] = y_pred
df_screen.to_csv("mlp_fusion_predicted_all_elements.csv", index=False)

print("✅ 已完成对 all_elements.csv 的能量预测，结果已保存为 'mlp_fusion_predicted_all_elements.csv'")

# === 设置统一 Nature 风格（再次声明以确保图形部分生效）===
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['figure.dpi'] = 150

# === 图1：R² 分布 ===
plt.figure(figsize=(8, 5))
sns.histplot(df_all["R2"], bins=15, kde=True, color="skyblue", label="All Runs", edgecolor='black')
sns.histplot(df_good["R2"], bins=10, kde=False, color="orange", label="R² > 0.75", edgecolor='black')
plt.xlabel("R² Score", fontsize=16, weight='bold')
plt.ylabel("Count", fontsize=16, weight='bold')
plt.title("Distribution of R² over 50 Runs (MLP Fusion)", fontsize=16, weight='bold')
plt.xticks(fontsize=14, weight='bold')
plt.yticks(fontsize=14, weight='bold')
plt.legend(fontsize=14)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)
plt.tight_layout()
plt.savefig("mlp_fusion_r2_distribution_nature.png", dpi=600)
plt.close()

# === 图2：MAE 分布 ===
plt.figure(figsize=(8, 5))
sns.histplot(df_all["MAE"], bins=15, kde=True, color="salmon", edgecolor='black')
plt.xlabel("MAE", fontsize=16, weight='bold')
plt.ylabel("Count", fontsize=16, weight='bold')
plt.title("Distribution of MAE over 50 Runs (MLP Fusion)", fontsize=16, weight='bold')
plt.xticks(fontsize=14, weight='bold')
plt.yticks(fontsize=14, weight='bold')
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)
plt.tight_layout()
plt.savefig("mlp_fusion_mae_distribution_nature.png", dpi=600)
plt.close()

# === 图3：预测能量直方图 ===
plt.figure(figsize=(8, 5))
sns.histplot(df_screen["Predicted_Energy"], bins=20, kde=True, color="mediumseagreen", edgecolor='black')
plt.xlabel("Predicted Energy (eV)", fontsize=16, weight='bold')
plt.ylabel("Count", fontsize=16, weight='bold')
plt.title("Distribution of Predicted Energy (All Elements)", fontsize=16, weight='bold')
plt.xticks(fontsize=14, weight='bold')
plt.yticks(fontsize=14, weight='bold')
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)
plt.tight_layout()
plt.savefig("mlp_fusion_pred_energy_distribution.png", dpi=600)
plt.close()

# === 图4：二维热图（元素 × 构型） ===
pivot_table = df_screen.pivot(index="Element", columns="SourceSite", values="Predicted_Energy")
plt.figure(figsize=(12, max(6, len(pivot_table) * 0.3)))
sns.heatmap(
    pivot_table, cmap="coolwarm", annot=False, linewidths=0.3, linecolor='gray',
    cbar_kws={'label': 'Predicted Energy (eV)'}
)
plt.xlabel("SourceSite", fontsize=16, weight='bold')
plt.ylabel("Element", fontsize=16, weight='bold')
plt.title("Predicted Energy Heatmap by Element and SourceSite", fontsize=16, weight='bold')
plt.xticks(rotation=45, ha='right', fontsize=14, weight='bold')
plt.yticks(rotation=0, fontsize=14, weight='bold')
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)
plt.tight_layout()
plt.savefig("mlp_fusion_pred_energy_heatmap.png", dpi=600)
plt.close()