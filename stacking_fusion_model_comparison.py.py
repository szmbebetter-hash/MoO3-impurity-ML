import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib as mpl

# ============ Nature 基线风格 ============
mpl.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 1.6,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "lines.linewidth": 1.8,
    "figure.dpi": 300,
})
sns.set_style("white")

nature_palette = ["#1f77b4", "#4c72b0", "#dd8452", "#55a868"]


# ============ 统一刻度 & 纵轴“放松一点” ============
def apply_nature_ticks(ax):
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        length=5,
        width=1.4,
        pad=5,
        bottom=True, left=True,
        top=False, right=False
    )
    return ax


def relax_y(ax, top_expand=0.25, bottom_expand=0.02, floor_zero=False):
    """
    让 y 轴上方、下方多一点空间，防止顶格：
    - top_expand: 上面多留多少比例空间（0.25 = 再加 25% 高度）
    - bottom_expand: 下面多留多少比例（一般 0.0–0.05）
    - floor_zero: 是否强制 y>=0（比如计数图）
    """
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin if ymax > ymin else 1.0

    new_ymin = ymin - bottom_expand * yr
    new_ymax = ymax + top_expand * yr

    if floor_zero:
        new_ymin = max(0.0, new_ymin)

    ax.set_ylim(new_ymin, new_ymax)
    return ax


# ============ Step 1 数据 ============
df = pd.read_csv("2.csv")
df = df[df["SourceSite"].str.contains("Site_3|Site_4")].copy()

# ============ Feature ============
df["EN_div_AR"] = df["Electronegativity"] / df["AtomicRadius"]
df["IE_div_AR"] = df["FirstIonizationEnergy"] / df["AtomicRadius"]
df["Group_x_Period"] = df["Group"] * df["period"]
df["EN_x_IE"] = df["Electronegativity"] * df["FirstIonizationEnergy"]

df["ReplacedAtomType"] = df["SourceSite"].apply(
    lambda x: 0 if ("Site_1" in x or "Site_2" in x) else 1
)
df["ReplacedAtomType_x_EN"] = df["ReplacedAtomType"] * df["Electronegativity"]
df["ReplacedAtomType_x_AR"] = df["ReplacedAtomType"] * df["AtomicRadius"]

feature_cols = [
    "Group","Electronegativity","AtomicRadius","FirstIonizationEnergy",
    "SiteTypeCode","period","block",
    "EN_div_AR","IE_div_AR","Group_x_Period","EN_x_IE",
    "ReplacedAtomType","ReplacedAtomType_x_EN","ReplacedAtomType_x_AR"
]

X = df[feature_cols].values
y = df["Energy (eV)"].values

# ============ Scaling ============
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ============ Base models ============
rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                           min_samples_split=3, max_features='sqrt',
                           random_state=42)
gbdt = GradientBoostingRegressor(n_estimators=300,
                                 learning_rate=0.05, max_depth=4,
                                 random_state=42)
svr = SVR(C=10, epsilon=0.2, gamma='scale')
mlp = MLPRegressor(hidden_layer_sizes=(128,64), activation='relu',
                   learning_rate_init=0.0006, max_iter=8000,
                   early_stopping=True, validation_fraction=0.1,
                   random_state=42)

base_learners = [("rf",rf),("gbdt",gbdt),("svr",svr),("mlp",mlp)]

fusion_models = {
    "Ridge": Ridge(alpha=1.0),
    "RF": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(C=1.0, epsilon=0.2),
    "MLP": MLPRegressor(hidden_layer_sizes=(64,), max_iter=3000, random_state=42)
}

results = []
all_preds = {}

for name, final_model in fusion_models.items():
    stack = StackingRegressor(
        estimators=base_learners,
        final_estimator=final_model,
        passthrough=False,
        n_jobs=-1
    )
    stack.fit(X_train, y_train)
    y_pred = stack.predict(X_val)

    results.append({
        "FusionModel": name,
        "MAE": mean_absolute_error(y_val, y_pred),
        "R2": r2_score(y_val, y_pred)
    })
    all_preds[name] = y_pred

results_df = pd.DataFrame(results)
results_df.to_csv("fusion_comparison_results.csv", index=False)


# ============ Figure 1 — R² Bar ============
plt.figure(figsize=(7,4.5))
ax = sns.barplot(
    data=results_df, x="FusionModel", y="R2",
    palette=nature_palette,
    width=0.38,
    edgecolor="black", linewidth=1.6
)
apply_nature_ticks(ax)
relax_y(ax, top_expand=0.2, bottom_expand=0.0, floor_zero=False)
ax.set_title("R² of Fusion Models")
ax.set_ylabel("R² Score")
ax.set_xlabel("")
for t in ax.get_xticklabels():
    t.set_weight("bold")
plt.tight_layout()
plt.savefig("fusion_R2_bar.png", dpi=300)
plt.close()


# ============ Figure 2 — MAE Bar ============
plt.figure(figsize=(7,4.5))
ax = sns.barplot(
    data=results_df, x="FusionModel", y="MAE",
    palette=nature_palette,
    width=0.38,
    edgecolor="black", linewidth=1.6
)
apply_nature_ticks(ax)
relax_y(ax, top_expand=0.2, bottom_expand=0.0, floor_zero=True)
ax.set_title("MAE of Fusion Models")
ax.set_ylabel("Mean Absolute Error")
ax.set_xlabel("")
for t in ax.get_xticklabels():
    t.set_weight("bold")
plt.tight_layout()
plt.savefig("fusion_MAE_bar.png", dpi=300)
plt.close()


# ============ Figure 3 — Pred vs True ============
for name, y_pred in all_preds.items():
    plt.figure(figsize=(6.0,6.0))
    ax = plt.gca()

    plt.scatter(
        y_val, y_pred,
        s=58, alpha=0.82,
        facecolor="#4c72b0",
        edgecolor="black",
        linewidth=1.3
    )

    lims = [min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())]
    plt.plot(lims, lims, "--", color="black", linewidth=2.0)

    plt.xlabel("DFT Energy (eV)")
    plt.ylabel("Predicted Energy (eV)")
    plt.title(f"{name} Prediction vs True")

    apply_nature_ticks(ax)
    relax_y(ax, top_expand=0.15, bottom_expand=0.05, floor_zero=False)
    plt.tight_layout()
    plt.savefig(f"{name}_pred_vs_true.png", dpi=300)
    plt.close()


# ============ Figure 4 — Residual Histogram ============
for name, y_pred in all_preds.items():
    residuals = y_val - y_pred

    plt.figure(figsize=(6.2,4.3))
    ax = plt.gca()

    sns.histplot(
        residuals, bins=18, kde=True,
        color="#55a868",
        edgecolor="black",
        linewidth=1.4
    )

    plt.title(f"{name} Residual Distribution")
    plt.xlabel("Residual (DFT − Prediction)")

    apply_nature_ticks(ax)
    # 残差分布是 Count，从 0 开始，所以强制 floor_zero=True
    relax_y(ax, top_expand=0.3, bottom_expand=0.0, floor_zero=True)
    plt.tight_layout()
    plt.savefig(f"{name}_residual_hist.png", dpi=300)
    plt.close()