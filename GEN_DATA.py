import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
N = 100  # Number of rows
distance = np.random.uniform(100, 5000, N)  # Distance in km
weight = np.random.uniform(1, 50, N)  # Weight in tons
vehicle_type = np.random.choice(['Truck', 'Train', 'Ship', 'Plane'], N)
fuel_efficiency = np.random.normal(5, 1, N)  # km/unit fuel

# Create DataFrame
df = pd.DataFrame({
    'Distance': distance,
    'Weight': weight,
    'Vehicle_Type': vehicle_type,
    'Fuel_Efficiency': fuel_efficiency
})

# Save to Excel
df.to_excel('sample_logistics_data.xlsx', index=False)
print("Excel file 'sample_logistics_data.xlsx' generated successfully!")