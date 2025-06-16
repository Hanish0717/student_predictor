import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load student data
data = pd.read_csv("data.csv")

# Convert Marks to Pass/Fail
data["Result"] = data["Marks"].apply(lambda x: 1 if x >= 16 else 0)

X = data[["Hours_Studied", "Attendance", "Assignments_Completed"]]
y = data["Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, "pass_fail_model.pkl")
print("✅ Model trained and saved.")
