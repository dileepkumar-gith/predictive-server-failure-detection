import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#load the data set
df = pd.read_csv('ml/data/Failure_Prediction.csv')
# define our feature columns and target columns
feature_columns = [ 
    "cpu_usage_percent", "ram_usage_percent", "disk_usage_percent",
    "network_usage_percent", "active_jobs", "idle_time_minutes",
    "power_consumption_watts", "temperature_celsius"
]
 
X= df[feature_columns]
Y=df["failure_risk"]

print("Features (X) shape:", X.shape)
print("Target (y) shape:", Y.shape)
print("\nUnique target values:", Y.unique()) #no.of unique values in the target(Y)

# Split into 80% training, 20% testing
# random_state=42 ensures we get the same split every time we run this (reproducibility)
# stratify=y ensures both train and test sets keep the same proportion of Low/Medium/High
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

print("\nTraining set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])

# Create the Random Forest model
# n_estimators=100 means it builds 100 individual decision trees and combines their votes
# random_state=42 again ensures reproducibility
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model - this is where the actual "learning" happens
print("\nTraining the model... (this may take a few seconds)")
model.fit(X_train, Y_train)
print("Training complete!")

# Use the trained model to predict on the TEST set (data it has never seen)
Y_pred = model.predict(X_test)

# Calculate overall accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"\nOverall Accuracy: {accuracy:.2%}")

# Detailed breakdown: precision, recall, F1-score for EACH class (Low/Medium/High)
print("\nDetailed Classification Report:")
print(classification_report(Y_test, Y_pred))

# Confusion matrix: shows exactly which classes get confused with which
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred, labels=["Low", "Medium", "High"]))

