import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_loader import load_data

@st.cache_data
def get_model_comparisons():
    """
    Computes comparative metrics natively replicating exact Notebook pipeline logic.
    Identical constraints: test_size=0.2, random_state=42, duplicates dropped BEFORE split, target log1p/expm1.
    """
    df = load_data()
    X = df[['car_age', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'brand']]
    y = df['selling_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_features = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
    num_features = ['car_age', 'km_driven']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_features),
            ('num', StandardScaler(), num_features)
        ])

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }

    results = []
    
    y_test_mean = np.mean(y_test)
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        ttr = TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1
        )
        
        ttr.fit(X_train, y_train)
        
        y_pred = ttr.predict(X_test)
        y_pred_train = ttr.predict(X_train)
        
        mae_test = mean_absolute_error(y_test, y_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
        r2_test = r2_score(y_test, y_pred)
        
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)
        
        results.append({
            'Model': name,
            'MAE': mae_test,
            'RMSE': rmse_test,
            'R2': r2_test,
            'MAE_Percentage': (mae_test / y_test_mean) * 100,
            'Train_MAE': mae_train,
            'Train_R2': r2_train
        })

    return pd.DataFrame(results), y_test_mean
