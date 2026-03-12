import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.interpolate import make_interp_spline


# ================= Nature 统一风格 =================
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 14,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 1.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "legend.frameon": False,
    "savefig.dpi": 600,
    "figure.dpi": 160
})
sns.set_style("white")


def apply_ticks(ax):
    ax.tick_params(
        axis="both", which="both",
        direction="in", length=5, width=1.4, pad=5,
        bottom=True, left=True,
        top=False, right=False
    )
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_weight("bold")
    return ax


def relax_y(ax, top_expand=0.22, bottom_expand=0.02, floor_zero=False):
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin
    ymin2 = ymin - yr * bottom_expand
    ymax2 = ymax + yr * top_expand
    if floor_zero:
        ymin2 = max(0, ymin2)
    ax.set_ylim(ymin2, ymax2)
    return ax


# ================= 数据 =================
df = pd.read_csv("2.csv")

df["EN_div_AR"] = df["Electronegativity"] / df["AtomicRadius"]
df["IE_div_AR"] = df["FirstIonizationEnergy"] / df["AtomicRadius"]
df["Group_x_Period"] = df["Group"] * df["period"]
df["EN_x_IE"] = df["Electronegativity"] * df["FirstIonizationEnergy"]
df["ReplacedAtomType"] = df["SourceSite"].apply(
    lambda x: 0 if ("Site_1" in x or "Site_2" in x) else 1
)
df["ReplacedAtomType_x_EN"] = df["ReplacedAtomType"] * df["Electronegativity"]
df["ReplacedAtomType_x_AR"] = df["ReplacedAtomType"] * df["AtomicRadius"]

features = [
    "Group","Electronegativity","AtomicRadius","FirstIonizationEnergy",
    "SiteTypeCode","period","block",
    "EN_div_AR","IE_div_AR","Group_x_Period","EN_x_IE",
    "ReplacedAtomType","ReplacedAtomType_x_EN","ReplacedAtomType_x_AR"
]

X = df[features].values
y = df["Energy (eV)"].values


# ================= 数据集划分 =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# ================= 模型与搜索网格 =================
models = {
    "RF": (
        RandomForestRegressor(random_state=42),
        {
            "n_estimators":[150,200,250],
            "max_depth":[8,10,12],
            "min_samples_split":[2,3,4],
            "max_features":['sqrt']
        }
    ),
    "SVR": (
        SVR(),
        {
            "C":[0.5,1.0,2.0],
            "epsilon":[0.1,0.2,0.3],
            "gamma":["auto","scale"]
        }
    ),
    "GBDT": (
        GradientBoostingRegressor(random_state=42),
        {
            "n_estimators":[150,200,250],
            "learning_rate":[0.05,0.1,0.15],
            "max_depth":[3,4,5],
            "subsample":[0.9,1.0]
        }
    ),
    "MLP": (
        MLPRegressor(max_iter=3000, early_stopping=True, validation_fraction=0.1, random_state=42),
        {
            "hidden_layer_sizes":[(64,),(64,32),(128,)],
            "activation":["relu"],
            "learning_rate_init":[0.001,0.005],
            "alpha":[0.0001,0.0005,0.001]
        }
    )
}


# ================= 训练与评估 =================
results=[]
pred_df=pd.DataFrame({"True":y_val})

for name,(model,grid_params) in models.items():
    grid=GridSearchCV(model,grid_params,scoring="neg_mean_absolute_error",cv=3,n_jobs=-1)
    grid.fit(X_train,y_train)
    best=grid.best_estimator_
    y_pred=best.predict(X_val)

    results.append({"Model":name,"MAE":mean_absolute_error(y_val,y_pred),"R2":r2_score(y_val,y_pred),"BestParams":grid.best_params_})
    pred_df[name]=y_pred


results_df=pd.DataFrame(results)
results_df.to_csv("multi_model_results_with_ReplacedAtom.csv",index=False)
pred_df.to_csv("predictions.csv",index=False)


# ================= 选最优模型 & CV =================
best_row=results_df.sort_values("MAE").iloc[0]
best_name=best_row["Model"]
best_params=best_row["BestParams"]

final_model=models[best_name][0].set_params(**best_params)
kf=KFold(n_splits=5,shuffle=True,random_state=42)
y_cv_pred=cross_val_predict(final_model,X_scaled,y,cv=kf)

sorted_idx_full=np.argsort(y)
y_true_sorted=y[sorted_idx_full]
y_cv_pred_sorted=y_cv_pred[sorted_idx_full]


# ================= 再训练获取误差 =================
best_model=models[best_name][0].set_params(**best_params)
best_model.fit(X_train,y_train)
y_best_pred=best_model.predict(X_val)
errors=y_best_pred-y_val
sorted_idx=np.argsort(y_val)


# ================= 图 1 — MAE 柱 =================
plt.figure(figsize=(7.2,4.6))
ax=sns.barplot(data=results_df,x="Model",y="MAE",
               palette="viridis",width=0.38,
               edgecolor="black",linewidth=1.6)
apply_ticks(ax); relax_y(ax,top_expand=0.22,floor_zero=True)
ax.set_title("MAE Comparison Across Models")
ax.set_ylabel("Mean Absolute Error"); ax.set_xlabel("")
plt.tight_layout(); plt.savefig("figure_1_mae_nature.png"); plt.close()


# ================= 图 2 — R² 柱 =================
plt.figure(figsize=(7.2,4.6))
ax=sns.barplot(data=results_df,x="Model",y="R2",
               palette="plasma",width=0.38,
               edgecolor="black",linewidth=1.6)
apply_ticks(ax); relax_y(ax,top_expand=0.22)
ax.set_title("R² Score Comparison Across Models")
ax.set_ylabel("R² Score"); ax.set_xlabel("")
plt.tight_layout(); plt.savefig("figure_2_r2_nature.png"); plt.close()


# ================= 图 3 — True vs Pred =================
plt.figure(figsize=(6.0,6.0))
ax=plt.gca()
plt.scatter(y_val,y_best_pred,s=65,alpha=0.82,
            facecolor="#4c72b0",edgecolor="black",linewidth=1.2)
lims=[min(y_val.min(),y_best_pred.min()),max(y_val.max(),y_best_pred.max())]
plt.plot(lims,lims,"k--",linewidth=2.0)

plt.xlabel("True Energy (eV)")
plt.ylabel("Predicted Energy (eV)")
plt.title(f"{best_name} Prediction vs True")

apply_ticks(ax); relax_y(ax,top_expand=0.15,bottom_expand=0.05)
plt.tight_layout(); plt.savefig("figure_3_true_vs_pred_nature.png"); plt.close()


# ================= 图 4 — 残差分布 =================
plt.figure(figsize=(6.4,4.2))
ax=plt.gca()
sns.histplot(errors,bins=24,kde=True,color="#55a868",
             edgecolor="black",linewidth=1.2)
plt.xlabel("Prediction Error (eV)")
plt.ylabel("Count")
plt.title(f"{best_name} Residual Distribution")
apply_ticks(ax); relax_y(ax,top_expand=0.28,floor_zero=True)
plt.tight_layout(); plt.savefig("figure_4_error_hist_nature.png"); plt.close()


# ================= 图 5 — 模型预测曲线 =================
# (5) 所有模型预测曲线 —— 点 + 样条平滑
plt.figure(figsize=(10, 6))
ax = plt.gca()

x = np.arange(len(sorted_idx))

for name in pred_df.columns:
    if name != "True":
        y_curve = pred_df[name].values[sorted_idx]

        # 样条平滑
        xs = np.linspace(x.min(), x.max(), 300)
        ys = make_interp_spline(x, y_curve, k=3)(xs)

        plt.plot(xs, ys, linewidth=2.8, label=name)
        plt.scatter(x, y_curve, s=28, edgecolor="black", color="black")

# True 曲线
true_curve = pred_df["True"].values[sorted_idx]
xs = np.linspace(x.min(), x.max(), 300)
ys = make_interp_spline(x, true_curve, k=3)(xs)
plt.plot(xs, ys, "--", color="black", linewidth=3.0, label="True")

plt.title("Predicted Energy Curves by Model", fontsize=18, weight="bold")
plt.xlabel("Sample Index (Sorted)", fontsize=16, weight="bold")
plt.ylabel("Energy (eV)", fontsize=16, weight="bold")

# === 加粗刻度 & 打开下/左刻度 ===
ax.tick_params(axis="both", which="both",
               direction="in", length=6, width=2,
               bottom=True, left=True, top=False, right=False)

for t in ax.get_xticklabels()+ax.get_yticklabels():
    t.set_fontsize(14)
    t.set_weight("bold")

# === 图例放在框内 ===
plt.legend(
    fontsize=14,
    frameon=False,
    loc="upper left",
    prop={"weight": "bold"},   # ← 字体加粗
    handlelength=2.8,          # ← 线段加长更好看
    borderpad=0.3,
)

# === 边框加粗 ===
for s in ax.spines.values():
    s.set_linewidth(2.2)

plt.tight_layout()
plt.savefig("figure_5_prediction_curves_nature_bold.png", dpi=600)
plt.close()


# ================= 图 6 — CV 曲线 =================
plt.figure(figsize=(10, 6))
ax = plt.gca()

x = np.arange(len(y_true_sorted))

# True
xs = np.linspace(x.min(), x.max(), 300)
ys = make_interp_spline(x, y_true_sorted, k=3)(xs)
plt.plot(xs, ys, "--", color="black", linewidth=3.0, label="True")

# CV Pred
xs = np.linspace(x.min(), x.max(), 300)
ys = make_interp_spline(x, y_cv_pred_sorted, k=3)(xs)
plt.plot(xs, ys, color="#d48806", linewidth=2.8, label=f"{best_name} (CV Predicted)")

plt.title("Cross-Validated Predicted Energy Curve", fontsize=18, weight="bold")
plt.xlabel("Sample Index (Sorted by True Energy)", fontsize=16, weight="bold")
plt.ylabel("Energy (eV)", fontsize=16, weight="bold")

# === 加粗刻度 ===
ax.tick_params(axis="both", which="both",
               direction="in", length=6, width=2,
               bottom=True, left=True, top=False, right=False)

for t in ax.get_xticklabels()+ax.get_yticklabels():
    t.set_fontsize(14)
    t.set_weight("bold")

# === 图例放在框内 ===
plt.legend(
    fontsize=14,
    frameon=False,
    loc="upper left",
    prop={"weight": "bold"},   # ← 字体加粗
    handlelength=2.8,          # ← 线段加长更好看
    borderpad=0.3,
)

# === 边框加粗 ===
for s in ax.spines.values():
    s.set_linewidth(2.2)

plt.tight_layout()
plt.savefig("figure_6_cv_curve_nature_bold.png", dpi=600)
plt.close()