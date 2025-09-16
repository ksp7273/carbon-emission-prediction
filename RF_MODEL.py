import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib  # For saving the model (install if needed: pip install joblib)

# Step 1: Generate synthetic data
np.random.seed(42)
N = 1000
distance = np.random.uniform(100, 5000, N)
weight = np.random.uniform(1, 50, N)  # tons
vehicle_type = np.random.choice(['Truck', 'Train', 'Ship', 'Plane'], N)
fuel_efficiency = np.random.normal(5, 1, N)  # km/unit fuel

base_factors = {'Truck': 2.5, 'Train': 0.5, 'Ship': 0.3, 'Plane': 10}  # Arbitrary emission factors

data = []
for i in range(N):
    vt = vehicle_type[i]
    factor = base_factors[vt]
    consumption = distance[i] / fuel_efficiency[i]
    co2 = consumption * factor * weight[i]
    data.append([distance[i], weight[i], vt, fuel_efficiency[i], co2])

df = pd.DataFrame(data, columns=['Distance', 'Weight', 'Vehicle_Type', 'Fuel_Efficiency', 'CO2_Emissions'])

# Add noise for realism
noise_std = 0.03 * df['CO2_Emissions'].std()
noise = np.random.normal(0, noise_std, N)
df['CO2_Emissions'] += noise
df['CO2_Emissions'] = df['CO2_Emissions'].clip(lower=0)

# Step 2: Engineer features
df['Consumption'] = df['Distance'] / df['Fuel_Efficiency']
enc = OneHotEncoder(sparse_output=False)
vt_enc = enc.fit_transform(df[['Vehicle_Type']])
vt_df = pd.DataFrame(vt_enc, columns=enc.get_feature_names_out(['Vehicle_Type']))
df_eng = pd.concat([df[['Distance', 'Weight', 'Fuel_Efficiency', 'Consumption']], vt_df], axis=1)
X = df_eng
y = df['CO2_Emissions']

# Step 3: Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_percent = (rmse / y_test.mean()) * 100
print(f'RMSE: {rmse:.2f} kg CO2, Relative RMSE: {rmse_percent:.2f}%')

# Save model and encoder for app use
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(enc, 'encoder.pkl')