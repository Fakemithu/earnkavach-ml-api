import pandas as pd
import numpy as np
import os

# ensure folder exists
os.makedirs("../data", exist_ok=True)

n = 1000

data = pd.DataFrame({
    "day_of_week": np.random.randint(0, 7, n),
    "hour_of_day": np.random.randint(6, 23, n),
    "avg_last_7_days": np.random.randint(300, 1200, n),
    "rainfall": np.random.uniform(0, 100, n),
    "aqi": np.random.randint(50, 300, n),
    "temperature": np.random.uniform(15, 45, n),
    "zone_demand_score": np.random.uniform(0.3, 1.0, n),
    "is_weekend": np.random.randint(0, 2, n),
    "working_hours": np.random.uniform(4, 12, n)
})

data["daily_income"] = (
    data["avg_last_7_days"] * 0.6 +
    data["zone_demand_score"] * 500 -
    data["rainfall"] * 2 -
    data["aqi"] * 1 +
    data["working_hours"] * 50 +
    np.random.normal(0, 50, n)
)

data.to_csv("../data/dataset.csv", index=False)

print("Dataset generated successfully!")