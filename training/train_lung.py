from sklearn.ensemble import RandomForestClassifier
import pickle
from training.split_data import split_lung_data

# Get split datasets
X_train_lung, X_test_lung, y_train_lung, y_test_lung = split_lung_data()

# Train AI model
lung_model = RandomForestClassifier(n_estimators=100, random_state=42)
lung_model.fit(X_train_lung, y_train_lung)

# Save trained model
pickle.dump(lung_model, open("model/lung_cancer_model.pkl", "wb"))

print("✅ Lung Cancer Model Saved Successfully!")