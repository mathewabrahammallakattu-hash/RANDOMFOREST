import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
data = pd.read_csv('Iris.csv')
label_encoder = LabelEncoder()
label_encoder.fit(data["Species"])

x=data.drop(["Species", "Id"], axis=1)
y=label_encoder.transform(data["Species"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100, "%")


# Save the trained model
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")