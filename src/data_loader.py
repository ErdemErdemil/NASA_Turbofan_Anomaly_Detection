import pandas as pd

def load_turbofan_data(file_path):
    """
    Loads the NASA turbofan engine dataset from the given file path.

    Parameters:
        file_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: The loaded dataset as a Pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path, sep=' ', header=None)
        #Remove empty columns that result from extra spaces
        data = data.loc[:, ~data.columns.duplicated()]
        data = data.dropna(axis=1, how='all')
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def check_data_integrity(df):
    """
    Checks for missing values and prints basic information about the DataFrame.

    Parameters:
        df (pd.DataFrame): The dataset to check.

    Returns:
        None
    """
    if df is None:
        print("DataFrame is None. Please load the data first.")
        return
    
    print("First 5 rows of the dataset:")
    print(df.head())
    
    print("\nDataFrame shape:")
    print(df.shape)
    
    print("\nMissing values in each column:")
    print(df.isnull().sum())
    
    print("\nData types of each column:")
    print(df.dtypes)

def rename_turbofan_columns(df):
    """
    Renames the columns of the turbofan dataset DataFrame with meaningful names.

    Parameters:
        df (pd.DataFrame): The dataset with default numeric column names.

    Returns:
        pd.DataFrame: The dataset with renamed columns.
    """
    if df is None:
        print("DataFrame is None. Cannot rename columns.")
        return None

    column_names = [
        'unit_number',  #Engine ID
        'time_in_cycles',  #Time cycle
        'operational_setting_1',
        'operational_setting_2',
        'operational_setting_3',
        'sensor_measurement_1',
        'sensor_measurement_2',
        'sensor_measurement_3',
        'sensor_measurement_4',
        'sensor_measurement_5',
        'sensor_measurement_6',
        'sensor_measurement_7',
        'sensor_measurement_8',
        'sensor_measurement_9',
        'sensor_measurement_10',
        'sensor_measurement_11',
        'sensor_measurement_12',
        'sensor_measurement_13',
        'sensor_measurement_14',
        'sensor_measurement_15',
        'sensor_measurement_16',
        'sensor_measurement_17',
        'sensor_measurement_18',
        'sensor_measurement_19',
        'sensor_measurement_20',
        'sensor_measurement_21'
    ]

    if df.shape[1] != len(column_names):
        print(f"Warning: The DataFrame has {df.shape[1]} columns but {len(column_names)} names provided.")
        print("Columns will not be renamed.")
        return df

    df.columns = column_names
    return df