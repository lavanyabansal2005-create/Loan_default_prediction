# Credit Risk Modeling & Loan Default Prediction

## Project Overview
This project builds an end-to-end machine learning pipeline to predict institutional loan defaults based on borrower profiles. It addresses the critical challenge of the **Accuracy Paradox** in highly imbalanced financial datasets by shifting the evaluation focus from raw accuracy to precision, recall, and F1-scores.

## Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy, Scipy (`spearmanr`)
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)

## Pipeline & Methodology

### 1. Exploratory Data Analysis (EDA) & Preprocessing
* Visualized numerical distributions using Kernel Density Estimates (KDE) and categorical frequencies via bar charts.
* Applied **One-Hot Encoding** (`drop_first=True`) to categorical variables to prevent multicollinearity (the dummy variable trap).

### 2. Feature Selection
* Conducted correlation analysis using a **Spearman Rank Correlation Matrix** to identify monotonic relationships and filter out redundant features or low-variance noise.

### 3. Handling Class Imbalance
The dataset exhibited a severe **84:16 class imbalance**, where models naturally favored predicting the majority class (No Default) to artificially inflate accuracy. This was mitigated using:
* **Algorithmic Weighting:** Utilizing `class_weight='balanced'` for Random Forest and `scale_pos_weight` for XGBoost to heavily penalize false negatives.
* **SMOTE (Synthetic Minority Over-sampling Technique):** Artificially balancing the training data prior to model fitting.

### 4. Modeling & Evaluation
Trained and evaluated three distinct classifier architectures:
* **Logistic Regression** (Baseline)
* **Random Forest Classifier**
* **XGBoost Classifier** (Champion Model)

**Evaluation Shift:** Because 84% baseline accuracy could be achieved by predicting '0' every time, models were evaluated strictly on their ability to detect the minority class using **Recall**, **Precision**, and the **F1-Score**. 

## Installation and Usage

1. Clone this repository:
   ```bash
   git clone [https://github.com/NV-2005/Loan_default_prediction.git](https://github.com/NV-2005/Loan_default_prediction.git)
   cd Loan_default_prediction
