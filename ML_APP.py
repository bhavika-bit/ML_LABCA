# app.py
import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

# EARTH ENGINE INITIALIZATION
try:
    ee.Initialize()
except Exception as e:
    st.error("Earth Engine not authenticated. Run ee.Authenticate() locally once before deploying.")
    st.stop()

# STREAMLIT UI
st.title("🌾 Crop Recommendation System with Satellite Data")

st.sidebar.header("📍 Field Coordinates")
lat_min = st.sidebar.number_input("Latitude (South/Bottom)", value=19.0)
lon_min = st.sidebar.number_input("Longitude (West/Left)", value=72.8)
lat_max = st.sidebar.number_input("Latitude (North/Top)", value=19.2)
lon_max = st.sidebar.number_input("Longitude (East/Right)", value=73.0)

aoi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

# MODIS LST
dataset = ee.ImageCollection("MODIS/061/MOD11A1") \
    .filterDate("2022-12-01", "2022-12-31") \
    .filterBounds(aoi)

lst_day = dataset.select("LST_Day_1km").mean().clip(aoi)
lst_celsius = lst_day.multiply(0.02).subtract(273.15)

mean_temp = lst_celsius.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=aoi,
    scale=1000
).getInfo()

if mean_temp and 'LST_Day_1km' in mean_temp:
    avg_temp = round(mean_temp['LST_Day_1km'], 2)
    st.success(f"🌡️ Avg Land Surface Temperature: {avg_temp} °C")
else:
    st.warning("Could not retrieve temperature for the selected area.")
    st.stop()

# Satellite Map
st.subheader("🛰️ Satellite LST View")
Map = geemap.Map()
Map.centerObject(aoi, 10)
Map.addLayer(lst_celsius, {
    'min': 10,
    'max': 45,
    'palette': [
        '040274','0502a3','235cb1','269db1','30c8e2',
        '3be285','3ae237','b5e22e','ffd611','ff6e08','ff0000'
    ]
}, 'LST °C')
Map.addLayer(aoi, {'color': 'black'}, 'AOI')
Map.to_streamlit()

# Soil & Weather Inputs
st.sidebar.header("🧪 Soil & Weather Info")
n = st.sidebar.number_input("Nitrogen (kg/ha)", min_value=0.0)
p = st.sidebar.number_input("Phosphorus (kg/ha)", min_value=0.0)
k = st.sidebar.number_input("Potassium (kg/ha)", min_value=0.0)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
ph = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0)

input_features = [n, p, k, avg_temp, humidity, ph, rainfall]

# Load ML Model
@st.cache_resource
def load_model():
    df = pd.read_csv("Crop_recommendation.csv")
    df.drop_duplicates(inplace=True)
    df['Crop'] = df['Crop'].map({
        'rice': 0, 'maize': 1, 'chickpea': 2, 'pigeonpeas': 3, 'mothbeans': 4,
        'mungbean': 5, 'blackgram': 6, 'lentil': 7, 'pomegranate': 8, 'banana': 9,
        'mango': 10, 'grapes': 11, 'watermelon': 12, 'muskmelon': 13, 'apple': 14,
        'orange': 15, 'papaya': 16, 'coconut': 17, 'cotton': 18, 'jute': 19, 'coffee': 20
    })
    df.dropna(subset=['Crop'], inplace=True)

    X = df[['N','P','K','temperature','humidity','ph','rainfall']]
    y = df['Crop']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=500)
    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=21, n_estimators=500)
    rf.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    voting = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb_model)], voting='hard')
    voting.fit(X_train, y_train)

    return voting, scaler

model, scaler = load_model()

# Prediction
if st.button("🚀 Recommend Crop"):
    input_scaled = scaler.transform(np.array(input_features).reshape(1, -1))
    pred = model.predict(input_scaled)[0]

    crop_mapping = {
        0: 'Rice', 1:'Maize', 2:'Chickpeas', 3:'Pigeon Peas', 4:'Moth Beans',
        5:'Mung Beans', 6:'Blackgram', 7:'Lentils', 8:'Pomegranates', 9:'Banana',
        10:'Mango', 11:'Grapes', 12:'Watermelon', 13:'Muskmelon', 14:'Apple',
        15:'Orange', 16:'Papaya', 17:'Coconut', 18:'Cotton', 19:'Jute', 20:'Coffee'
    }

    st.success(f"✅ Recommended Crop: **{crop_mapping[pred]}**")
