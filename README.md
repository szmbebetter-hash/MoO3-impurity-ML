# Machine Learning Prediction of Impurity Stability in α-MoO3

This repository contains datasets and example Python scripts used in the study:

**"Machine learning assisted prediction of impurity stability across multiple configurations in α-MoO3."**

The purpose of this repository is to provide representative data and demonstration scripts for the machine learning workflow used in this work.

---

# Repository Structure

## Dataset

The datasets correspond to different atomic configurations (sites) considered in the study.

| File | Description |
|-----|-------------|
| site_1.csv | Dataset for configuration Site_1 |
| site_2.csv | Dataset for configuration Site_2 |
| site_3.csv | Dataset for configuration Site_3 |
| site_4.csv | Dataset for configuration Site_4 |
| site_5.csv | Dataset for configuration Site_5 |
| site_6.csv | Dataset for configuration Site_6 |
| element_screening_pool_filled_final.csv | Element screening dataset used for prediction |

Each dataset contains elemental descriptors and the corresponding calculated energy values.

---

## Python Scripts

The provided Python scripts demonstrate the machine learning workflow used in this study.

| Script | Function |
|------|-----------|
| train_and_evaluate_models.py | Training and evaluation of multiple ML models |
| ml_model_comparison_with_cross_validation.py | Model comparison with cross-validation |
| stacking_fusion_model_comparison.py | Stacking ensemble model comparison |
| stacking_mlp_fusion_stability_and_screening.py | Stability analysis and element screening |
| plot_strategy_comparison.py | Visualization of model comparison results |

These scripts illustrate the feature engineering, model training, validation, and visualization procedures used in the study.

---

# Machine Learning Methods

The following machine learning models were explored:

- Random Forest (RF)
- Gradient Boosting Decision Trees (GBDT)
- Support Vector Regression (SVR)
- Multi-Layer Perceptron (MLP)
- Stacking Ensemble Models

Model performance was evaluated using:

- Mean Absolute Error (MAE)
- R² Score
- Cross-validation analysis

---

# Data Availability

The datasets and example scripts used in this work are available in this repository.

Additional scripts used for intermediate analysis and data processing are available from the corresponding author upon reasonable request.

---

# Requirements

Python packages required to run the scripts include:
