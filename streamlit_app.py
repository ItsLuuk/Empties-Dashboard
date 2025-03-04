import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import xgboost as xgb
from prophet import Prophet

st.title("Empties Dashboard")
st.write("Inzicht in de pickups (pallets) en de leeggoed die terugkomt, zodat beslissingen snel gemaakt kunnen worden")

# Laad de gegevens uit het Excel-bestand
bestandspad = "/workspaces/Empties-Dashboard/Hoogvliet Forecast.xlsx"
df = pd.read_excel(bestandspad)

df.columns = df.columns.str.strip()
df['Emballagestroom'] = df['Emballagestroom'].replace(0, df[df['Emballagestroom'] != 0]['Emballagestroom'].mean())
df = df.drop('Datum', axis=1, errors='ignore')

if 'Klant' not in df.columns:
    st.error("De kolom 'Klant' is niet gevonden in het Excel-bestand.")
    st.stop()

geselecteerde_klant = st.selectbox("Selecteer een klant", df['Klant'].unique())
gefilterde_df = df[df['Klant'] == geselecteerde_klant].copy()

# Variabelen uitleg
st.markdown("**Rol van Dag en Week in de Forecast:**")
st.write("- **Dag**: Houdt rekening met dagelijkse patronen binnen een week (bijvoorbeeld pieken op maandag en dalen op zondag).")
st.write("- **Week**: Helpt bij het identificeren van bredere trends en seizoensinvloeden over meerdere weken.")

feature_cols_all = ['Dag', 'Week', 'HL', 'HL 1 week', 'HL 2 weken', 'HL 3 weken', 'HL 4 weken']
feature_cols_no_week_day = ['HL', 'HL 1 week', 'HL 2 weken', 'HL 3 weken', 'HL 4 weken']

missing_cols = [col for col in feature_cols_all if col not in gefilterde_df.columns]
if missing_cols:
    st.error(f"De volgende kolommen ontbreken in de dataset: {missing_cols}")
    st.stop()

for col in feature_cols_all:
    gefilterde_df[col] = pd.to_numeric(gefilterde_df[col], errors='coerce')
gefilterde_df = gefilterde_df.dropna()

st.write("Gegevens voor geselecteerde klant:")
st.dataframe(gefilterde_df)

def train_and_evaluate(X, y, feature_names, model_name, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X)
    mape = mean_absolute_percentage_error(y, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    importances = getattr(model, 'feature_importances_', None)
    if importances is None:  # Voor modellen zonder feature_importances_
        perm_importances = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        importances = perm_importances.importances_mean
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return model, y_pred, mape, rmse, feature_importance_df

# Modellen initialiseren
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "LightGBM": lgb.LGBMRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42),
    "Prophet": Prophet()
}

# Model selectie
selected_model_name = st.selectbox("Selecteer een model", list(models.keys()))
selected_model = models[selected_model_name]

if selected_model_name == "Prophet":
    # Weeknummers omzetten naar datums
    start_year = 2023  # Vervang door het juiste jaar
    start_date = pd.to_datetime(f'{start_year}-01-01')  # Start van het jaar
    
    def week_to_date(week_num):
        return start_date + pd.Timedelta(weeks=week_num - 1)  # Aanname: week 1 start op 1 januari

    prophet_df = gefilterde_df[['Week', 'Emballagestroom']].copy()
    prophet_df['ds'] = prophet_df['Week'].apply(week_to_date)
    prophet_df = prophet_df[['ds', 'Emballagestroom']].rename(columns={'Emballagestroom': 'y'})

    selected_model.fit(prophet_df)
    future = selected_model.make_future_dataframe(periods=0, freq='W')
    forecast = selected_model.predict(future)
    y_pred = forecast['yhat'].values
    feature_importance_df = pd.DataFrame()
    mape = mean_absolute_percentage_error(gefilterde_df['Emballagestroom'], y_pred) * 100
    rmse = np.sqrt(mean_squared_error(gefilterde_df['Emballagestroom'], y_pred))
else:
    selected_model, y_pred, mape, rmse, feature_importance_df = train_and_evaluate(
        gefilterde_df[feature_cols_all], gefilterde_df['Emballagestroom'], feature_cols_all, selected_model_name, selected_model
    )
    mape = mean_absolute_percentage_error(gefilterde_df['Emballagestroom'], y_pred) * 100
    rmse = np.sqrt(mean_squared_error(gefilterde_df['Emballagestroom'], y_pred))

# Resultaten weergeven
st.subheader(f"Model: {selected_model_name}")
st.write(f"MAPE = {mape:.2f}%, RMSE = {rmse:.2f}")

if not feature_importance_df.empty:
    st.subheader("Feature Importance")
    st.dataframe(feature_importance_df)

gefilterde_df[f'Forecast_{selected_model_name}'] = y_pred
st.write("DataFrame met Forecast:")
st.dataframe(gefilterde_df)
