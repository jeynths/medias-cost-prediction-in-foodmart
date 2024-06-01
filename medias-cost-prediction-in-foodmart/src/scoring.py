import mlflow
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

logged_model = 'runs:/4f67706f9a80477e9f7f65ac894bd47f/models'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

data = pd.read_csv(r'medias-cost-prediction-in-foodmart\\input\\customer_campaign_cost_prediction.csv')

# label encoding of the objects

for i in data.columns:
    if data.dtypes[i] == object:
        lb_make = LabelEncoder()
        data[i] = lb_make.fit_transform(data[i])

data = data.drop(["cost", "avg_cars_at home(approx)", "prepared_food",
                  "avg_cars_at home(approx)", "salad_bar", "store_state",
                  "coffee_bar"], axis=1)

# Predict on a Pandas DataFrame.

result = loaded_model.predict(pd.DataFrame(data))

data["predicted_results"] = result
print(data)
