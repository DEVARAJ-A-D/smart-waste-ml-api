import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# 🔹 Step 1: Create synthetic dataset

data = []
locations = ["home", "public", "market"]

for i in range(300):

    capacity = np.random.randint(50, 300)
    avg_waste = np.random.uniform(1, 10)
    hours = np.random.randint(1, 24)
    location = np.random.choice(locations)

    # 🔥 Base fill logic
    fill = avg_waste * hours

    # 🔥 Location impact (real-world behavior)
    if location == "market":
        fill *= 1.5
    elif location == "public":
        fill *= 1.2

    # 🔥 Add randomness (noise)
    fill += np.random.uniform(-10, 10)

    # 🔥 Normalize using capacity
    fill_percentage = (fill / capacity) * 100

    # 🔥 Clamp values between 0 and 100
    fill_percentage = max(0, min(fill_percentage, 100))

    data.append([capacity, avg_waste, hours, location, fill_percentage])

# 🔹 Step 2: Convert to DataFrame

df = pd.DataFrame(data, columns=[
    "capacity", "avgWaste", "hours", "location", "fill"
])

# 🔥 Save dataset (IMPORTANT)
df.to_csv("dataset.csv", index=False)
print("Dataset saved as dataset.csv")

# 🔹 Step 3: Encode location (categorical → numeric)

df["location"] = df["location"].map({
    "home": 0,
    "public": 1,
    "market": 2
})

# 🔹 Step 4: Prepare features and target

X = df[["capacity", "avgWaste", "hours", "location"]]
y = df["fill"]

# 🔹 Step 5: Train model

model = LinearRegression()
model.fit(X, y)

# 🔹 Step 6: Save model

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")