import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

def plot_engine_cycle_distribution(df):
    """
    Plots the distribution of cycle counts for each engine unit.

    Parameters:
        df (pd.DataFrame): The dataset containing unit_number and time_in_cycles.

    Returns:
        None
    """
    if df is None:
        print("DataFrame is None. Cannot plot.")
        return
    
    cycle_counts = df.groupby('unit_number')['time_in_cycles'].max()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(cycle_counts, bins=20, kde=True)
    plt.title("Distribution of Engine Cycle Lifespans")
    plt.xlabel("Number of Cycles")
    plt.ylabel("Number of Engines")
    plt.show()


def plot_sensor_distributions(df, sensors):
    """
    Plots the distribution of specified sensor measurements.

    Parameters:
        df (pd.DataFrame): The dataset containing sensor measurements.
        sensors (list): List of sensor column names to plot.

    Returns:
        None
    """
    if df is None:
        print("DataFrame is None. Cannot plot.")
        return
    
    for sensor in sensors:
        if sensor not in df.columns:
            print(f"Sensor {sensor} not found in DataFrame columns.")
            continue
        
        plt.figure(figsize=(10, 6))
        sns.histplot(df[sensor], bins=50, kde=True)
        plt.title(f"Distribution of {sensor}")
        plt.xlabel(sensor)
        plt.ylabel("Frequency")
        plt.show()

def identify_constant_sensors(df):
    """
    Identifies sensor columns that have zero variance (constant values).

    Parameters:
        df (pd.DataFrame): The dataset.

    Returns:
        list: List of sensor column names with constant values.
    """
    if df is None:
        print("DataFrame is None. Cannot check sensors.")
        return []

    constant_sensors = []
    for col in df.columns:
        if col.startswith('sensor_measurement'):
            if df[col].nunique() == 1:
                constant_sensors.append(col)

    if constant_sensors:
        print("Constant sensors found:")
        for sensor in constant_sensors:
            print(f"- {sensor}")
    else:
        print("No constant sensors found.")

    return constant_sensors

def drop_constant_sensors(df, constant_sensors):
    """
    Drops the specified constant sensor columns from the dataset.

    Parameters:
        df (pd.DataFrame): The dataset.
        constant_sensors (list): List of sensor column names to drop.

    Returns:
        pd.DataFrame: The dataset without the constant sensors.
    """
    if df is None:
        print("DataFrame is None. Cannot drop sensors.")
        return None

    if not constant_sensors:
        print("No constant sensors to drop.")
        return df

    df_dropped = df.drop(columns=constant_sensors)
    print(f"Dropped {len(constant_sensors)} constant sensors.")
    return df_dropped

def plot_operational_settings_distribution(df):
    """
    Plots the distributions of the operational setting columns.

    Parameters:
        df (pd.DataFrame): The dataset.

    Returns:
        None
    """
    if df is None:
        print("DataFrame is None. Cannot plot operational settings.")
        return

    operational_settings = [col for col in df.columns if col.startswith('operational_setting')]

    for setting in operational_settings:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[setting], bins=50, kde=True)
        plt.title(f"Distribution of {setting}")
        plt.xlabel(setting)
        plt.ylabel("Frequency")
        plt.show()

def drop_columns(df, columns_to_drop):
    """
    Drops the specified columns from the dataset.

    Parameters:
        df (pd.DataFrame): The dataset.
        columns_to_drop (list): List of column names to drop.

    Returns:
        pd.DataFrame: The dataset without the specified columns.
    """
    if df is None:
        print("DataFrame is None. Cannot drop columns.")
        return None

    if not columns_to_drop:
        print("No columns to drop.")
        return df

    df_dropped = df.drop(columns=columns_to_drop)
    print(f"Dropped {len(columns_to_drop)} columns: {columns_to_drop}")
    return df_dropped

def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap for sensor and operational setting columns.

    Parameters:
        df (pd.DataFrame): The dataset.

    Returns:
        None
    """
    if df is None:
        print("DataFrame is None. Cannot compute correlation.")
        return

    #Select numeric columns only (exclude unit_number and time_in_cycles)
    cols = [col for col in df.columns if col not in ['unit_number', 'time_in_cycles']]
    
    corr_matrix = df[cols].corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap of Sensors and Operational Settings")
    plt.show()

def plot_sensor_boxplots(df, sensors):
    """
    Plots boxplots for specified sensor columns to visualize outliers.

    Parameters:
        df (pd.DataFrame): The dataset.
        sensors (list): List of sensor column names to plot.

    Returns:
        None
    """
    if df is None:
        print("DataFrame is None. Cannot plot boxplots.")
        return

    for sensor in sensors:
        if sensor not in df.columns:
            print(f"Sensor {sensor} not found in DataFrame columns.")
            continue
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[sensor])
        plt.title(f"Boxplot of {sensor}")
        plt.xlabel(sensor)
        plt.show()

def compute_rul(df):
    """
    Computes Remaining Useful Life (RUL) for each row in the dataset.

    Parameters:
        df (pd.DataFrame): The dataset containing unit_number and time_in_cycles.

    Returns:
        pd.DataFrame: The dataset with a new RUL column.
    """
    if df is None:
        print("DataFrame is None. Cannot compute RUL.")
        return None

    max_cycles = df.groupby('unit_number')['time_in_cycles'].transform('max')
    df['RUL'] = max_cycles - df['time_in_cycles']
    print("RUL column added to dataset.")
    return df

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
importances_df = None

def train_random_forest_and_get_feature_importance(df):
    """
    Trains a Random Forest Regressor on the dataset and returns feature importance.

    Parameters:
        df (pd.DataFrame): The dataset including RUL.

    Returns:
        pd.DataFrame: DataFrame of features and their importance.
    """
    if df is None:
        print("DataFrame is None. Cannot train model.")
        return None

    # Features and target
    feature_cols = [col for col in df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
    X = df[feature_cols]
    y = df['RUL']

    #Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    #Feature importance
    importances = model.feature_importances_
    importances_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    #Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=importances_df)
    plt.title("Feature Importance from Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    return importances_df

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_rf_and_evaluate(df):
    """
    Trains a Random Forest Regressor, evaluates performance, and plots feature importance.

    Parameters:
        df (pd.DataFrame): The dataset with RUL.

    Returns:
        pd.DataFrame: Feature importance DataFrame.
    """
    if df is None:
        print("DataFrame is None. Cannot train model.")
        return None

    feature_cols = [col for col in df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
    X = df[feature_cols]
    y = df['RUL']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) 
    model.fit(X_train, y_train)

    #Predictions and metrics
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    #Feature importance
    importances = model.feature_importances_
    importances_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=importances_df)
    plt.title("Feature Importance from Random Forest (Optimized)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    return importances_df

def add_rolling_features(df, window=5):
    """
    Adds rolling mean and rolling std for each sensor column based on time_in_cycles.

    Parameters:
        df (pd.DataFrame): The dataset.
        window (int): The window size for rolling calculation.

    Returns:
        pd.DataFrame: The dataset with new rolling features.
    """
    if df is None:
        print("DataFrame is None. Cannot add rolling features.")
        return None

    #Sort to ensure correct rolling computation
    df = df.sort_values(by=['unit_number', 'time_in_cycles'])
    
    sensor_cols = [col for col in df.columns if col.startswith('sensor_measurement')]

    for sensor in sensor_cols:
        df[f'{sensor}_roll_mean'] = (
            df.groupby('unit_number')[sensor]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f'{sensor}_roll_std'] = (
            df.groupby('unit_number')[sensor]
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )
    
    print(f"Rolling features (mean & std, window={window}) added for each sensor.")
    return df

def train_lightgbm_and_evaluate(df):
    """
    Trains a LightGBM regressor and evaluates performance.

    Parameters:
        df (pd.DataFrame): The dataset with RUL.

    Returns:
        pd.DataFrame: Feature importance DataFrame.
    """
    if df is None:
        print("DataFrame is None. Cannot train model.")
        return None

    feature_cols = [col for col in df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
    X = df[feature_cols]
    y = df['RUL']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"LightGBM R² Score: {r2:.4f}")
    print(f"LightGBM MAE: {mae:.4f}")
    print(f"LightGBM RMSE: {rmse:.4f}")

    importances_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=importances_df)
    plt.title("Feature Importance from LightGBM")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    return importances_df

def train_xgboost_and_evaluate(df):
    """
    Trains an XGBoost regressor and evaluates performance.

    Parameters:
        df (pd.DataFrame): The dataset with RUL.

    Returns:
        pd.DataFrame: Feature importance DataFrame.
    """
    if df is None:
        print("DataFrame is None. Cannot train model.")
        return None

    feature_cols = [col for col in df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
    X = df[feature_cols]
    y = df['RUL']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"XGBoost R² Score: {r2:.4f}")
    print(f"XGBoost MAE: {mae:.4f}")
    print(f"XGBoost RMSE: {rmse:.4f}")

    importances_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=importances_df)
    plt.title("Feature Importance from XGBoost")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    return importances_df

def compute_prediction_errors(df, model):
    """
    Computes prediction errors using the given model.

    Parameters:
        df (pd.DataFrame): The dataset with RUL.
        model: The trained model.

    Returns:
        pd.DataFrame: DataFrame with an 'error' column.
    """
    feature_cols = [col for col in df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
    X = df[feature_cols]
    y = df['RUL']
    
    df['predicted_RUL'] = model.predict(X)
    df['error'] = np.abs(df['RUL'] - df['predicted_RUL'])
    
    print("Prediction errors computed.")
    return df


def compute_anomaly_threshold(df):
    """
    Computes anomaly threshold using IQR.

    Parameters:
        df (pd.DataFrame): DataFrame with 'error' column.

    Returns:
        float: anomaly threshold
    """
    Q1 = df['error'].quantile(0.25)
    Q3 = df['error'].quantile(0.75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR
    print(f"Anomaly threshold determined: {threshold:.4f}")
    return threshold


def plot_prediction_error_distribution(df, threshold):
    """
    Plots the prediction error distribution and threshold.

    Parameters:
        df (pd.DataFrame): DataFrame with 'error' column.
        threshold (float): The anomaly threshold.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['error'], bins=50, kde=True)
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
    plt.title("Prediction Error Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def train_rf_and_return_model(df):
    """
    Trains Random Forest model and returns the trained model and the dataset with predictions.

    Parameters:
        df (pd.DataFrame): Dataset

    Returns:
        model: Trained Random Forest model
        pd.DataFrame: Dataset with prediction error
    """
    feature_cols = [col for col in df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
    X = df[feature_cols]
    y = df['RUL']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    #Make predictions for all data
    df['predicted_RUL'] = model.predict(X)
    df['error'] = np.abs(df['RUL'] - df['predicted_RUL'])

    r2 = r2_score(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

    print(f"RF (Anomaly Model) R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    return model, df
