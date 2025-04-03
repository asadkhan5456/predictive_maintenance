import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

# ------------------------------------------------------------
# Step 1: ETL Functions (Data Sourcing, Cleaning, Feature Engineering)
# ------------------------------------------------------------

def load_data(file_path, column_names):
    """
    Load data from a whitespace-separated file.
    """
    df = pd.read_csv(file_path, sep='\s+', header=None, names=column_names)
    return df

def clean_data(df):
    """
    Clean the dataset by checking for missing values and handling them properly.
    """
    if df.isnull().sum().sum() > 0:
        df = df.fillna(method='ffill')
    return df

def feature_engineering(df):
    """
    Perform some feature engineering on the DataFrame.
    """
    sensor_columns = [col for col in df.columns if col.startswith('sensor')]
    df['sensor_avg'] = df[sensor_columns].mean(axis=1)
    return df

def process_data():
    """
    Orchestrates the ETL process:
    - Loads raw training and test data.
    - Cleans and processes the data.
    - Saves the processed data to a new folder.
    """
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data')
    processed_dir = os.path.join(data_dir, 'processed')
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    column_names = ['unit_number', 'time', 'op_setting1', 'op_setting2', 'op_setting3'] + \
                   [f'sensor{i}' for i in range(1, 22)]
    
    # Process training data
    train_file = os.path.join(data_dir, 'train_FD001.txt')
    train_df = load_data(train_file, column_names)
    train_df = clean_data(train_df)
    train_df = feature_engineering(train_df)
    train_processed_path = os.path.join(processed_dir, 'train_processed.csv')
    train_df.to_csv(train_processed_path, index=False)
    print("Processed training data saved to:", train_processed_path)
    
    # Process test data
    test_file = os.path.join(data_dir, 'test_FD001.txt')
    test_df = load_data(test_file, column_names)
    test_df = clean_data(test_df)
    test_df = feature_engineering(test_df)
    test_processed_path = os.path.join(processed_dir, 'test_processed.csv')
    test_df.to_csv(test_processed_path, index=False)
    print("Processed test data saved to:", test_processed_path)

# ------------------------------------------------------------
# Step 2: EDA (Exploratory Data Analysis)
# ------------------------------------------------------------

sns.set(style="whitegrid")

# Define the path to the processed training data
def run_eda():
    """
    Run Exploratory Data Analysis on processed training data.
    """
    processed_data_path = os.path.join(os.getcwd(), "data", "processed", "train_processed.csv")
    df = pd.read_csv(processed_data_path)
    
    # Display first few rows and summary statistics
    print("First 5 rows of the training data:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())
    
    # Plot distribution of sensor_avg
    plt.figure(figsize=(8, 5))
    sns.histplot(df['sensor_avg'], bins=50, kde=True)
    plt.title('Distribution of Sensor Average')
    plt.xlabel('Sensor Average')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(os.getcwd(), "data", "processed", "sensor_avg_distribution.png"))
    plt.show()
    
    # Plot time series of sensor_avg for a specific engine unit (e.g., unit 1)
    unit_id = 1
    unit_data = df[df['unit_number'] == unit_id]
    plt.figure(figsize=(10, 6))
    plt.plot(unit_data['time'], unit_data['sensor_avg'], marker='o', linestyle='-', label='Sensor Average')
    plt.title(f'Sensor Average Over Time for Unit {unit_id}')
    plt.xlabel('Time')
    plt.ylabel('Sensor Average')
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), "data", "processed", f"unit{unit_id}_sensor_avg_timeseries.png"))
    plt.show()

# ------------------------------------------------------------
# Step 3: Labeling and Model Building (Step 4 in our original plan)
# ------------------------------------------------------------

def add_RUL_labels(df):
    """
    For training data: Compute RUL as the difference between the maximum time (failure point) and current time.
    """
    df = df.copy()
    max_times = df.groupby('unit_number')['time'].max().reset_index().rename(columns={'time': 'max_time'})
    df = df.merge(max_times, on='unit_number', how='left')
    df['RUL'] = df['max_time'] - df['time']
    df.drop(columns=['max_time'], inplace=True)
    return df

def add_test_RUL_labels(test_df, rul_file_path):
    """
    For test data: Compute RUL using provided final RUL and the time gap.
    """
    test_df = test_df.copy()
    max_times_test = test_df.groupby('unit_number')['time'].max().reset_index().rename(columns={'time': 'max_time'})
    test_df = test_df.merge(max_times_test, on='unit_number', how='left')
    
    rul_df = pd.read_csv(rul_file_path, sep='\s+', header=None, names=['final_RUL'])
    final_RUL_array = rul_df['final_RUL'].values
    
    unique_units = np.sort(test_df['unit_number'].unique())
    if len(unique_units) != len(final_RUL_array):
        raise ValueError("Mismatch between number of test units and RUL values provided.")
    rul_mapping = dict(zip(unique_units, final_RUL_array))
    
    test_df['provided_RUL'] = test_df['unit_number'].map(rul_mapping)
    test_df['RUL'] = test_df['provided_RUL'] + (test_df['max_time'] - test_df['time'])
    test_df.drop(columns=['max_time', 'provided_RUL'], inplace=True)
    return test_df

def build_and_train_model():
    """
    Constructs RUL labels, trains a baseline Random Forest model, and evaluates it.
    Then, it saves the trained model to disk.
    """
    # Load processed training data
    train_processed_path = os.path.join(os.getcwd(), 'data', 'processed', 'train_processed.csv')
    train_df = pd.read_csv(train_processed_path)
    train_df = add_RUL_labels(train_df)
    print("Training data with RUL labels (first 5 rows):")
    print(train_df[['unit_number', 'time', 'RUL']].head())
    
    # Load processed test data and add RUL labels using provided file
    test_processed_path = os.path.join(os.getcwd(), 'data', 'processed', 'test_processed.csv')
    test_df = pd.read_csv(test_processed_path)
    rul_file_path = os.path.join(os.getcwd(), 'data', 'RUL_FD001.txt')
    test_df = add_test_RUL_labels(test_df, rul_file_path)
    print("\nTest data with computed RUL labels (first 5 rows):")
    print(test_df[['unit_number', 'time', 'RUL']].head())
    
    # Prepare features and target for modeling
    feature_cols = ['op_setting1', 'op_setting2', 'op_setting3', 'sensor_avg']
    target_col = 'RUL'
    X = train_df[feature_cols]
    y = train_df[target_col]
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print("\nBaseline Random Forest RMSE on validation set: {:.2f}".format(rmse))
    
    # Save the trained model
    model_dir = os.path.join(os.getcwd(), "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)
    print("Trained model saved to:", model_path)
# ------------------------------------------------------------
# Main Execution: Run all steps sequentially
# ------------------------------------------------------------
if __name__ == '__main__':
    # Step 1: Process and save the data
    process_data()
    
    # Step 2: Run EDA on the processed training data
    run_eda()
    
    # Step 3 (Step 4 in our original plan): Label the data and build the predictive model
    build_and_train_model()



