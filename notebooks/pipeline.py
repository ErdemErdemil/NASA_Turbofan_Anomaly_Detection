
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("YOUR_DATA_FILE.csv")  # <--- DATA DOSYANIN ADINI GÜNCELLE

# Feature engineering (örneğin rolling mean/std varsa buraya ekle)
# ...

# Model
feature_cols = [col for col in df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
X = df[feature_cols]
y = df['RUL']

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)
df['predicted_RUL'] = model.predict(X)
df['error'] = np.abs(df['RUL'] - df['predicted_RUL'])

# Threshold
Q1 = df['error'].quantile(0.25)
Q3 = df['error'].quantile(0.75)
IQR = Q3 - Q1
threshold = Q3 + 1.5 * IQR

df['is_anomaly'] = df['error'] > threshold

# Export anomalies
df[df['is_anomaly']].to_csv("anomalies_detected.csv", index=False)

# Plot & save
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='time_in_cycles', y='error', hue='is_anomaly', palette={True: 'red', False: 'blue'}, legend='brief')
plt.title("Prediction Error over Cycles with Anomalies Highlighted")
plt.savefig("cycle_anomaly_plot.png")
plt.close()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='unit_number', y='error', hue='is_anomaly', palette={True: 'red', False: 'blue'}, legend='brief')
plt.title("Prediction Error by Unit Number with Anomalies Highlighted")
plt.savefig("unit_anomaly_plot.png")
plt.close()

print("Pipeline completed. Files saved.")
