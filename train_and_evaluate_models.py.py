import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib as mpl

# ========= Nature 基线风格 =========
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
    "lines.linewidth": 1.9,
    "figure.dpi": 300,
})
sns.set_style("white")

nature_palette = ["#1f77b4", "#4c72b0", "#dd8452", "#55a868"]

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

def relax_y(ax, top_expand=0.22, bottom_expand=0.02, floor_zero=False):
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin if ymax > ymin else 1.0
    new_ymin = ymin - bottom_expand * yr
    new_ymax = ymax + top_expand * yr
    if floor_zero:
        new_ymin = max(0.0, new_ymin)
    ax.set_ylim(new_ymin, new_ymax)
    return ax


# ===== Step 1: Load Data =====
df = pd.read_csv("2.csv")
df = df[df["SourceSite"].str.contains("Site_3|Site_4")].copy()

# ===== Step 2: Feature Engineering =====
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

# ===== Step 3: Split & Scale =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===== Step 4: Models =====
models = {
    "RF": (
        RandomForestRegressor(random_state=42),
        {"n_estimators":[200],"max_depth":[10],
         "min_samples_split":[3],"max_features":["sqrt"]}
    ),
    "SVR": (
        SVR(),
        {"C":[1.0],"epsilon":[0.2],"gamma":["auto"]}
    ),
    "GBDT": (
        GradientBoostingRegressor(random_state=42),
        {"n_estimators":[200],"learning_rate":[0.1],"max_depth":[4]}
    ),
    "MLP": (
        MLPRegressor(max_iter=3000, early_stopping=True,
                     validation_fraction=0.1, random_state=42),
        {"hidden_layer_sizes":[(64,32)],
         "activation":["relu"],
         "learning_rate_init":[0.001]}
    )
}

# ===== Step 5: Train, Tune, Evaluate =====
results=[]
pred_df=pd.DataFrame({"True":y_val})

for name,(model,param_grid) in models.items():
    grid = GridSearchCV(model,param_grid,scoring="neg_mean_absolute_error",
                        cv=3,n_jobs=-1)
    grid.fit(X_train,y_train)
    best = grid.best_estimator_
    y_pred = best.predict(X_val)

    results.append({
        "Model":name,
        "MAE":mean_absolute_error(y_val,y_pred),
        "R2":r2_score(y_val,y_pred),
        "BestParams":grid.best_params_
    })
    pred_df[name]=y_pred

results_df=pd.DataFrame(results)
results_df.to_csv("multi_model_results_with_ReplacedAtom.csv",index=False)
pred_df.to_csv("predictions.csv",index=False)


# ===== FIGURE 1 — MAE Bar =====
plt.figure(figsize=(7,4.6))
ax = sns.barplot(
    data=results_df, x="Model", y="MAE",
    palette=nature_palette, width=0.38,
    edgecolor="black", linewidth=1.6
)
apply_nature_ticks(ax)
relax_y(ax, top_expand=0.22, floor_zero=True)
ax.set_title("MAE Comparison Across Models")
ax.set_ylabel("Mean Absolute Error")
ax.set_xlabel("")
for t in ax.get_xticklabels(): t.set_weight("bold")
plt.tight_layout()
plt.savefig("figure_1_mae.png",dpi=300)
plt.close()


# ===== FIGURE 2 — R² Bar =====
plt.figure(figsize=(7,4.6))
ax = sns.barplot(
    data=results_df, x="Model", y="R2",
    palette=nature_palette, width=0.38,
    edgecolor="black", linewidth=1.6
)
apply_nature_ticks(ax)
relax_y(ax, top_expand=0.22)
ax.set_title("R² Score Comparison Across Models")
ax.set_ylabel("R² Score")
ax.set_xlabel("")
for t in ax.get_xticklabels(): t.set_weight("bold")
plt.tight_layout()
plt.savefig("figure_2_r2.png",dpi=300)
plt.close()


# ===== FIGURE 3 — True vs Predicted =====
best_row = results_df.sort_values("MAE").iloc[0]
best_name = best_row["Model"]

best_model = models[best_name][0].set_params(**best_row["BestParams"])
best_model.fit(X_train,y_train)
y_best_pred = best_model.predict(X_val)

plt.figure(figsize=(6.0,6.0))
ax = plt.gca()

plt.scatter(
    y_val, y_best_pred,
    s=58, alpha=0.82,
    facecolor="#4c72b0",
    edgecolor="black",
    linewidth=1.3
)

lims=[min(y_val.min(),y_best_pred.min()),
      max(y_val.max(),y_best_pred.max())]
plt.plot(lims,lims,"--",color="black",linewidth=2.0)

plt.xlabel("True Energy (eV)")
plt.ylabel("Predicted Energy (eV)")
plt.title(f"True vs Predicted Energy ({best_name})")

apply_nature_ticks(ax)
relax_y(ax, top_expand=0.18, bottom_expand=0.04)
plt.tight_layout()
plt.savefig("figure_3_true_vs_pred.png",dpi=300)
plt.close()


# ===== FIGURE 4 — Error Histogram =====
errors = y_best_pred - y_val

plt.figure(figsize=(6.4,4.4))
ax = plt.gca()

sns.histplot(
    errors, bins=24, kde=True,
    color="#55a868",
    edgecolor="black",
    linewidth=1.4
)

plt.title(f"Prediction Error Distribution ({best_name})")
plt.xlabel("Prediction Error (eV)")

apply_nature_ticks(ax)
relax_y(ax, top_expand=0.30, floor_zero=True)
plt.tight_layout()
plt.savefig("figure_4_error_hist.png",dpi=300)
plt.close()


# ===== FIGURE 5 — Prediction Curves =====
plt.figure(figsize=(9.0,5.2))
ax = plt.gca()

sorted_idx = np.argsort(y_val)

for name in pred_df.columns:
    if name != "True":
        plt.plot(pred_df[name].values[sorted_idx],
                 label=name, linewidth=2.0)

plt.plot(pred_df["True"].values[sorted_idx],
         "k--", linewidth=2.2, label="True")

plt.title("Predicted Energy Curves by Model")
plt.ylabel("Energy (eV)")
plt.xlabel("Sample Index (Sorted)")
plt.legend(frameon=False)

apply_nature_ticks(ax)
relax_y(ax, top_expand=0.18)
plt.tight_layout()
plt.savefig("figure_5_prediction_curves.png",dpi=300)
plt.close()