import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import numpy as np

# Flatten data
def flatten_data(data):
    flat_data = []
    for entry in data:
        flat_entry = {}
        flat_entry.update(entry['Specs'])
        flat_entry['Price'] = entry['Price']
        # Convert Location to string
        flat_entry['Location'] = ', '.join(entry['Location'])
        flat_data.append(flat_entry)
    return flat_data

# Load data and train model
def train_models(data):
    # Flatten data and preprocess
    df = pd.DataFrame(flatten_data(data))

    # Convert 'Price' to numeric, coerce errors to NaN
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # Convert 'Area' column to numeric type
    df['Area'] = pd.to_numeric(df['Area'], errors='coerce')

    # Remove rows with NaN in the target variable
    df.dropna(subset=['Price'], inplace=True)

    # Split data
    X = df.drop('Price', axis=1)
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps
    numeric_features = ['Area']
    categorical_features = ['Rooms', 'Floor', 'Location']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean', missing_values=np.nan)),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Define models
    models = {
        'randomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'linearRegression': LinearRegression(),
        'supportVectorRegression': SVR()
    }

    model_pipelines = {}
    metrics_scores = {}

    # Train models
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_pipelines[name] = pipeline
        metrics_scores[name] = {
            "MAE": mae,
            "MSE": mse,
            "R2": r2
        }

    return df, model_pipelines, metrics_scores


def query_model(df, model, Location=None, Rooms=None, Floor=None, Area_range=None):
    # Filter DataFrame based on location parameters
    filtered_df = df.copy()

    # Remove column 'Price per square meter'
    filtered_df.drop('Price per square meter', axis=1, inplace=True)

    if Location:
        # Split the input into keywords
        keywords = Location.split()
        # Filter rows that contain all keywords
        for keyword in keywords:
            filtered_df = filtered_df[filtered_df['Location'].str.contains(keyword, case=False, na=False)]
    
    # Filter DataFrame based on number of rooms
    if Rooms and Rooms != 'Wszystkie opcje':
        filtered_df = filtered_df[filtered_df['Rooms'] == Rooms]
    
    # # Filter DataFrame based on floor
    if Floor and Floor != 'Wszystkie opcje':
        filtered_df = filtered_df[filtered_df['Floor'] == Floor]

    # Filter DataFrame based on area
    if Area_range is not None:
        filtered_df = filtered_df[(filtered_df['Area'] >= Area_range[0]) & (filtered_df['Area'] <= Area_range[1])]
    
    if filtered_df.empty:
        st.error("No matching properties found.")
        return None
    
    # Prepare features from the filtered data
    X_filtered = filtered_df.drop(['Price'], axis=1)

    # Predict prices
    predicted_prices = model.predict(X_filtered)
    filtered_df['Predicted Price'] = np.ceil(predicted_prices)

    # Reorder columns
    location_column = filtered_df.pop('Location')
    filtered_df['Location'] = location_column

    return filtered_df

# Encode categorical columns
def encode_categorical_columns(df, columns):
    mappings = {}
    for col in columns:
        df[col] = df[col].astype('category')
        mappings[col] = dict(enumerate(df[col].cat.categories))
        df[col] = df[col].cat.codes
    return df, mappings

# Decode categorical columns
def decode_categorical_columns(df, mappings):
    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)
    return df
    
