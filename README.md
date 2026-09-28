# 🚗 CarWise: Used Car Price Prediction & Market Analysis

> A Data Warehousing and Mining (DWM) mini project that covers the full data-mining pipeline, from cleaning and exploratory analysis to K-Means clustering and regression, and delivers it through an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75)

---

## 📌 Overview

Used-car prices depend on many factors: how old the car is, how much it has been driven, its fuel type, transmission, ownership history and who is selling it. Estimating a fair price by hand is difficult.

**CarWise** uses historical CarDekho data to discover price-related patterns and provides a data-driven **estimated selling price** for a user's car details. The project is not only an ML model. Its value lies in the complete DWM pipeline: data understanding, cleaning, transformation, EDA, clustering, regression, evaluation and visualization.

## 🎯 Objectives

- Understand and preprocess the CarDekho used-car dataset
- Identify relationships between car attributes and selling price
- Visualize market trends and descriptive statistics
- Apply **K-Means clustering** to discover meaningful groups of used cars
- Build and compare **regression models** for price prediction
- Select the best model using MAE, RMSE and R²
- Build an interactive dashboard for analysis and prediction
- Document assumptions, limitations and future improvements

## ✨ Features

| Page | What it does |
| --- | --- |
| 🏠 **Home / Overview** | Project summary, dataset KPIs and the DWM workflow |
| 💰 **Price Predictor** | Form for car details, input validation, predicted selling price |
| 📊 **Market Analysis** | Interactive Plotly charts of price vs. fuel, transmission, seller type, owner, year and kilometres driven |
| 🧩 **Car Segmentation** | K-Means cluster sizes, characteristics and visualization |
| 💡 **Insights** | Key findings and takeaways from the analysis |

## 🗂️ Dataset

- **Source:** CarDekho used-car dataset
- **Size:** 4,340 records originally, **3,577** after removing duplicate rows
- **Original attributes:** `name`, `year`, `selling_price` (target), `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`

**Features used for modelling**

| Feature | Type | Description |
| --- | --- | --- |
| `car_age` | Numerical | Derived as `2020 - year` |
| `km_driven` | Numerical | Distance driven |
| `fuel` | Categorical | Petrol, Diesel, CNG, LPG, Electric |
| `seller_type` | Categorical | Individual, Dealer, Trustmark Dealer |
| `transmission` | Categorical | Manual / Automatic |
| `owner` | Categorical | Ownership category |
| `brand` | Categorical | Extracted from the car name, which avoids one-hot encoding thousands of unique model names |
| `selling_price` | Numerical | **Target variable** |

## 🔄 DWM Workflow

```
Data Understanding → Cleaning → Transformation → EDA → K-Means Clustering
        → Regression Modelling → Model Selection → Saved Pipeline → Streamlit Dashboard
```

1. **Data Understanding:** shape, data types, unique values, duplicates and missing values
2. **Data Cleaning:** duplicate rows investigated and removed; outliers inspected before any decision
3. **Data Transformation:** categorical encoding, scaling for K-Means, and the derived `car_age` feature
4. **EDA:** price distribution and relationships between price and every feature
5. **Data Mining:** K-Means clustering on scaled numerical features
6. **Regression:** train/test split and multiple regressors compared
7. **Model Selection:** chosen using evaluation metrics
8. **Packaging:** model and preprocessing saved together as one pipeline
9. **Dashboard:** Streamlit app with Plotly visualizations

## 🧩 Clustering (K-Means)

K-Means with **K = 3** groups the cars into three market segments:

| Segment | Description |
| --- | --- |
| **Older Budget** | Older cars with lower prices |
| **Newer Mid-Range** | Newer cars in the middle price band |
| **Premium** | High-priced cars |

Features were scaled before clustering because K-Means is distance-based and would otherwise be dominated by large-valued features such as `km_driven`. Clustering is an analytical feature of the project and is not part of the price-prediction calculation.

## 🤖 Price Prediction

The selling price is a continuous value, so this is a **regression** problem. Candidate models:

- Linear Regression (baseline)
- Decision Tree Regressor
- Random Forest Regressor

**Selected model: Random Forest Regressor** (R² ≈ **0.5890** on the test set). It handles non-linear relationships between the features and price better than the other candidates. The fitted model and its preprocessing steps are saved as a single pipeline, so the **exact same preprocessing** is applied at training and at prediction time.

**Evaluation metrics:** MAE, RMSE and R².

## 🛠️ Tech Stack

| Layer | Tools |
| --- | --- |
| Language | Python |
| Data handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine learning | Scikit-learn |
| Dashboard | Streamlit |
| Model persistence | Joblib |
| Development | Jupyter Notebook, VS Code |
| Version control | Git, GitHub |

## CarWise/
├── data/
│   └── car_details.csv
├── models/
│   └── car_price_random_forest.pkl
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing_eda.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_price_prediction.ipynb
├── src/
│   └── prediction.py
├── .streamlit/
│   └── config.toml
├── app.py
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/ankita01209/dwm-project.git
cd dwm-project
```

**2. Create and activate a virtual environment (recommended)**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

## 🧪 Input Validation & Testing

The Price Predictor validates input before running the model.

| Test case | Expected result |
| --- | --- |
| Valid car details | Price is predicted and displayed |
| Negative kilometres | Validation error, no prediction |
| Unrealistic manufacturing year | Clear boundary message |
| Missing required field | User is prompted to complete it |
| Unsupported category | Input rejected |
| Model file missing | Controlled error message, no crash |
| Same input run twice | Identical result (reproducible) |


## 🔮 Future Scope

- Retrain periodically on more recent listings
- Add richer attributes such as engine, mileage, power, location and insurance status
- Show a price range or prediction interval instead of a single value
- Add explainability with SHAP
- Deploy the dashboard online
- Add a budget-based car recommendation feature
