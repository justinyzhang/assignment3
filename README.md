
#  Assignment3: Predicting Fast-Growing Firms – A Data-Driven Classification Approach

This repository presents a supervised learning framework for identifying fast-growing firms, using historical firm-level panel data from Bisnode (2010–2015). The objective is to build a robust prediction pipeline that not only performs well statistically, but also aligns with business-relevant decision-making under asymmetric costs.

## Problem Motivation

Fast-growing firms are key drivers of job creation, innovation, and productivity growth. Identifying such firms early can help investors, policymakers, and support organizations allocate resources more effectively. However, fast growth is rare, often nonlinear, and difficult to detect using simple heuristics.

This project treats **fast growth prediction** as a binary classification problem with:
- **Class imbalance**
- **Business-oriented misclassification costs**
- **Industry heterogeneity**

## Target Definition (Label Engineering)

We define a firm as **fast-growing** if, between 2012 and 2014:
- Revenue increased by at least **44%**, and
- At least one of the following holds:
  - Employment increased by ≥ 21% **and** personnel costs grew ≤ 32%
  - Profits increased ≥ 32% or turned from negative to positive
  - Fixed assets increased by ≥ 21%

This definition balances financial expansion and real operational scaling, based on principles from corporate finance (e.g., sustainable growth, cost control, capital accumulation). Simpler alternatives (e.g. revenue-only or profit-only rules) were tested but found less informative.

## Methodology Overview

We train three models on 16,619 firm-year observations:
1. **Logistic Regression** – as a linear baseline
2. **Random Forest** – to capture non-linearities and interactions
3. **Gradient Boosting** – for improved accuracy and calibration

Key steps include:
- Data cleaning, imputation, and one-hot encoding of categorical variables
- 80/20 train-test split, stratified by label
- 5-fold cross-validation on training data
- Prediction of **probabilities**, not just labels

## Business-Driven Classification

Instead of defaulting to a threshold of 0.5, we define a **loss function**:

- False Positive (Type I error): $1
- False Negative (Type II error): $5

This reflects a typical business scenario: missing a high-growth firm is more costly than over-investing in a low-growth one. We **optimize the threshold** for each model to minimize expected loss, then evaluate performance on the holdout set.

## Key Results

| Model               | ROC AUC | F1 Score | Threshold | Expected Loss |
|--------------------|---------|----------|-----------|----------------|
| Logistic Regression | 0.668   | 0.405    | 0.168     | $2315          |
| Random Forest       | 0.688   | 0.428    | 0.208     | $2320          |
| **Gradient Boosting**   | **0.677**   | **0.436**    | **0.208**     | **$2269**          |

Gradient Boosting was selected as the final model due to superior performance under both statistical and cost-sensitive criteria.

## Industry-Specific Evaluation
We apply the selected model separately to:
- **Manufacturing firms** (n ≈ 5,400)
- **Service firms** (n ≈ 11,200)

Each subgroup has its own optimal threshold and expected loss, reflecting differences in growth predictability:
- **Manufacturing AUC:** 0.6699, Loss: $797
- **Services AUC:** 0.6851, Loss: $1473

This supports a **segmented deployment strategy**.

## Visualizations

The repository includes:
- Threshold vs Expected Loss curve (to justify classification strategy)
- Confusion Matrix (with recall and precision)
- ROC curves for model comparison
- Industry-specific bar plots (Expected Loss, AUC)


## Summary

This project demonstrates how to integrate:
- Economic logic (target definition)
- Machine learning techniques (classification)
- Business priorities (cost-sensitive evaluation)

---



