import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

# ensure model folder exists
os.makedirs("../models", exist_ok=True)

# load data
data = pd.read_csv("../data/dataset.csv")

X = data.drop("daily_income", axis=1)
y = data["daily_income"]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# evaluate
preds = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))

# save model
joblib.dump(model, "../models/income_model.pkl")

print("Model trained and saved!")