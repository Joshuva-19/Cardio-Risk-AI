import pandas as pd

df = pd.read_csv("health_data.csv")
df["age"] = (df["age"] / 365).astype(int)
df["bmi"] = df["weight"] / ((df["height"]/100)**2)
df = df.rename(columns={
    "ap_hi": "systolic_bp",
    "ap_lo": "diastolic_bp"
})
# print(df.head())
# print(df.shape)
# print(df.columns)
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
# df.fillna(df.mean(), inplace=True)
df = df.drop("id", axis=1)
X = df.drop("cardio", axis=1)  # input
y = df["cardio"]               # output
# print(X.head())
# print(y.head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# print(X_train.shape)
# print(X_test.shape)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# # Model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# print("Actual:", y_test.values[:10])
# print("Predicted:", y_pred[:10])

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# from sklearn.tree import DecisionTreeClassifier
# dt_model = DecisionTreeClassifier(random_state=42)
# dt_model.fit(X_train, y_train)
# y_pred_dt = dt_model.predict(X_test)
# from sklearn.metrics import accuracy_score

# print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
# from sklearn.metrics import confusion_matrix

# print(confusion_matrix(y_test, y_pred_dt))
# from sklearn.metrics import classification_report

# print(classification_report(y_test, y_pred_dt))
