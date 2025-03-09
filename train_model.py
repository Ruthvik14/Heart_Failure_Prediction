import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import shap
import joblib

df = pd.read_csv("heart.csv")

print(df.isnull().sum()) 

numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(15, 8))
df[numeric_columns].boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot for Outlier Detection")
plt.show()

categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

df_encoded.info()

plt.figure(figsize=(12,8))
sns.heatmap(df_encoded.corr(), annot=True, cmap="autumn", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

sns.countplot(x='HeartDisease', data=df_encoded)
plt.title("Target Variable Distribution")
plt.show()

X= df_encoded.drop(columns=['HeartDisease'])
y= df_encoded['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

#Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

rf_y_pred = rf_model.predict(X_test_scaled)

#XGB Model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)

xgb_y_pred = xgb_model.predict(X_test_scaled)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

print("RandomForestClassifier Accuracy: ", accuracy_score(y_test, rf_y_pred))
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("Classification Report:\n", classification_report(y_test, rf_y_pred))

print("XGBoost Accuracy:", accuracy_score(y_test, xgb_y_pred))
print("Classification Report:\n", classification_report(y_test, xgb_y_pred))

explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test_scaled)

shap.summary_plot(shap_values, X_test_scaled)

joblib.dump(rf_model, "heart_failure_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(df_encoded.columns, "feature_names.pkl")
joblib.dump(rf_accuracy, "accuracy.pkl")

print("Model and scaler saved successfully!")

