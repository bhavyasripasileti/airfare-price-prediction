import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

def generate_synthetic_data(n=1000):
    airlines = ['Indigo', 'Air India', 'SpiceJet', 'Vistara', 'GoAir']
    sources = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
    destinations = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
    stops = [0, 1, 2]
    
    data = []
    for _ in range(n):
        airline = random.choice(airlines)
        source = random.choice(sources)
        destination = random.choice([d for d in destinations if d != source])
        date = (datetime.now() + timedelta(days=random.randint(1, 90))).strftime('%Y-%m-%d')
        stop = random.choice(stops)
        price = random.randint(3000, 20000) + stop * 2000
        
        data.append([airline, source, destination, date, stop, price])
    
    df = pd.DataFrame(data, columns=['Airline', 'Source', 'Destination', 'Date', 'Stops', 'Price'])
    return df

df = generate_synthetic_data()

df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df.drop(columns=['Date'], inplace=True)

encoder = OneHotEncoder(drop='first', sparse_output=False)
categorical_features = ['Airline', 'Source', 'Destination']
categorical_encoded = encoder.fit_transform(df[categorical_features])
categorical_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_features))

df = pd.concat([df.drop(columns=categorical_features), categorical_df], axis=1)

scaler = StandardScaler()
X = df.drop(columns=['Price'])
Y = df['Price']
X_scaled = scaler.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, Y_train)

def predict_price(airline, source, destination, stops, day, month, model_choice='Random Forest'):
    input_data = pd.DataFrame([[airline, source, destination, stops, day, month]],
                               columns=['Airline', 'Source', 'Destination', 'Stops', 'Day', 'Month'])
    input_encoded = encoder.transform(input_data[categorical_features])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_features))
    input_final = pd.concat([input_data.drop(columns=categorical_features), input_encoded_df], axis=1)
    input_scaled = scaler.transform(input_final)
    
    prediction = models[model_choice].predict(input_scaled)[0]
    return f"Predicted Airfare: ₹{prediction:.2f}"


def run_app():
    st.title("Airfare Price Prediction")
    airline = st.selectbox("Airline", ['Indigo', 'Air India', 'SpiceJet', 'Vistara', 'GoAir'])
    source = st.selectbox("Source", ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'])
    destination = st.selectbox("Destination", [d for d in ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'] if d != source])
    date = st.date_input("Travel Date")
    stops = st.selectbox("Stops", [0, 1, 2])
    day = date.day
    month = date.month
    
    model_choice = st.selectbox("Select Model", list(models.keys()))
    prediction = predict_price(airline, source, destination, stops, day, month, model_choice)
    
    st.write(prediction)

