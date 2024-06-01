# Importing Libraries

import configparser as configpar
import warnings
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
import xgboost as xg
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ydata_profiling import ProfileReport
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# Pandas Config

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Importing Config files and other dependant variables from text file

configParser = configpar.RawConfigParser()
configFilePath = r'medias-cost-prediction-in-foodmart\\src\\configuration.txt'
configParser.read(configFilePath)
input_path = configParser.get('input', 'path')
output_path = configParser.get('output', 'path')
input_file = configParser.get('input', 'input_file_name')
target_variable = configParser.get('input', 'target_variable')

# Initiating ML flow and pulling configuration details from previous instance

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# mlflow.set_tracking_uri(r"sqlite:///mlflow.db")

experiment_name = "Default"
current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
experiment_id = current_experiment['experiment_id']

print(experiment_id)

# Reading the Dataframe from Config variables

df = pd.read_csv(input_path + input_file + ".csv")

df_copy = df.copy()

df = df.sample(n=50000, random_state=0)

print(df.shape)

# Generating EDA using Pandas - Ydata profiling

report = ProfileReport(df, minimal=True)

report.to_file(output_path + "EDA_Report.html")

# Removing unique column from the data

for i in df.columns:
    if len(df[i]) == df[i].nunique():
        df.drop([i], axis=1, inplace=True)

# converting numerical values into objects when levels are less than 10

for i in df.columns:
    if df.dtypes[i] != object:
        if df[i].nunique() == 2 or df[i].nunique() <= 10:
            df[i] = df[i].astype('object')

# label encoding of the objects

for i in df.columns:
    if df.dtypes[i] == object:
        lb_make = LabelEncoder()
        df[i] = lb_make.fit_transform(df[i])

# Initiating Target and Predictors

x = df.drop(target_variable, axis=1)

y = df[target_variable]


# From Observed Insights from Data - removing columns with multicollinearity

def calc_vif(x_val):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = x_val.columns
    vif["VIF"] = [variance_inflation_factor(x_val.values, val) for val in range(x_val.shape[1])]

    return vif.sort_values(ascending=False, by="VIF")


VIF_value = calc_vif(x)

# print(VIF_value)

x = x.drop(
    ["avg_cars_at home(approx)", "prepared_food",
     "avg_cars_at home(approx)", "salad_bar", "store_state",
     "coffee_bar"], axis=1)

# Performing Train and Test Split

x_train, x_test, y_train, y_test = train_test_split(x, y.astype(int),
                                                    stratify=y.astype(int),
                                                    test_size=0.3)

# ML FLow Process triggering

# datetime object containing current date and time
current_timestamp = datetime.now()

idx = current_timestamp

MLFLOW_RUN_NAME = f"run_{idx}"

# Model Initialization

linear_regressor = LinearRegression()
random_forest_regressor = RandomForestRegressor(n_estimators=200, random_state=0)
xgb_regressor = xg.XGBRegressor(objective='reg:linear',
                                n_estimators=10, seed=123)
# svm_regressor = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

# Ensemble Learning

ensemble_regressor = VotingRegressor([('lr', linear_regressor),
                                      ('rf', random_forest_regressor),
                                      ('xg', xgb_regressor)])

ensemble_regressor_model = ensemble_regressor.fit(x_train, y_train)

y_pred = ensemble_regressor_model.predict(x_test)

# Metrics Computation

r_squared_value = metrics.r2_score(y_test, y_pred)
mse_value = metrics.mean_squared_error(y_test, y_pred)
rmse_value = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
mae_value = metrics.mean_absolute_error(y_test, y_pred)

# writing metrics to sheet
df_met = pd.DataFrame({'Run ID': [MLFLOW_RUN_NAME],
                       'No.of Training Records': [len(x_train)],
                       'No.of Testing Records': [len(x_test)],
                       'R Squared Error': [r_squared_value],
                       'MSE Value': [mse_value],
                       'RMSE Value': [rmse_value],
                       'MAE Value': [mae_value]
                       })

print(df_met)

# Metrics file to output csv

df_met.to_csv(output_path + "model_metrics.csv", index=False)

# Model Predicted results to Output folder

predicted_cost_df = pd.DataFrame(y_pred, columns=["Predicted_cost"]).reset_index(drop=True)

actual_cost_df = pd.DataFrame(y_test).reset_index(drop=True)

x_test = pd.DataFrame(x_test).reset_index(drop=True)

df_final_pred = pd.concat([x_test, actual_cost_df, predicted_cost_df], axis=1)

df_final_pred.to_csv(output_path + "model_predicted_results.csv", index=False, header=True)


ml_flow_version = ""

with mlflow.start_run(experiment_id=experiment_id, run_name=MLFLOW_RUN_NAME) as run:

    # Logging Metrics in MLFlow
    mlflow.log_metric("No.of.Training.Records", len(x_train))
    mlflow.log_metric("No.of.Testing.Records", len(x_test))
    mlflow.log_metric("R squared value", r_squared_value)
    mlflow.log_metric("MSE Value", float(mse_value))
    mlflow.log_metric("RMSE Value", float(rmse_value))
    mlflow.log_metric("MAE Value", float(mae_value))

    # Logging model as artifact

    mlflow.sklearn.log_model(ensemble_regressor_model, artifact_path="models")

    mlflow.end_run()
