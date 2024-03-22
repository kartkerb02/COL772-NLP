import os
import sys
import json
import pandas as pd
import joblib

# from sklearn.metrics import accuracy_score, classification_report

# Check if the correct number of arguments are provided
if len(sys.argv) != 4:
    print("Usage: python predict_model.py <model_path> <test_data_path> <output_path>")
    sys.exit(1)

# Set paths from command-line arguments
model_dir = sys.argv[1]
model_path = os.path.join(model_dir, "model.pkl")
test_data_path = sys.argv[2]
output_path = sys.argv[3]

# Load the trained model
model = joblib.load(model_path)

# Load test data
with open(test_data_path, 'r') as f:
    test_data = json.load(f)
test_df = pd.DataFrame(test_data)
X_test = test_df['text']

# Make predictions
predictions = model.predict(X_test)

# Save predictions
with open(output_path, 'w') as f:
    for prediction in predictions:
        f.write(prediction + '\n')
print("Predictions saved at:", output_path)


# # Display classification report
# print("\nClassification Report:")
# print(classification_report(y_valid_new, predictions))
