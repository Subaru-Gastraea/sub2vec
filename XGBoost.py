import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

import os
import sys

# Add the parent directory to sys.path
# Executing path: ddx-on-ehr/models/sub2vec/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))     # ddx-on-ehr/models/
from XGB_Classifier import XGBoostClassifierWrapper

# Save model in "./save_model" directory
# Save classification report and confusion matrix in "./result" directory
if __name__ == '__main__':
    model_input = pd.read_csv('model_input.csv')
    num_classes  = 6

    # Prepare features and labels
    train_df = model_input[model_input["train_or_test"] == "train"]
    test_df = model_input[model_input["train_or_test"] == "test"]
    X_train = train_df.drop(columns=["subGraphID", "label", "train_or_test"])
    X_test = test_df.drop(columns=["subGraphID", "label", "train_or_test"])
    y_train = train_df["label"]
    y_test = test_df["label"]

    model_path = pathlib.Path("./save_model")
    model_path.mkdir(parents=True, exist_ok=True)
    save_model_path = model_path / "xgboost_model.json"
    if save_model_path.exists():
        print(f"Model already exists at {save_model_path}.")
        # Load the model using the wrapper
        model = XGBoostClassifierWrapper(model_path=save_model_path)
        model.load()
    else:
        print(f"Model does not exist at {save_model_path}. Training a new model.")

        # Train XGBoost model using the wrapper
        model = XGBoostClassifierWrapper(model_path=save_model_path, num_class=num_classes, eval_metric='mlogloss', objective='multi:softmax')
        model.fit(X_train, y_train)

        # Save the model
        model.save()

    # Evaluate performance
    accuracy, report, conf_matrix = model.evaluate(X_test, y_test)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", conf_matrix)

    # Save reports
    pathlib.Path("result").mkdir(parents=True, exist_ok=True)
    with open("result/classification_report.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    labels = [str(i) for i in range(num_classes)]
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("result/confusion_matrix.png")