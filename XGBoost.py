import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pathlib

# Save model in "./save_model" directory
# Save classification report and confusion matrix in "./result" directory
if __name__ == '__main__':
    model_input = pd.read_csv('model_input.csv')

    # Encode labels
    le = LabelEncoder()
    le.fit_transform(model_input["label"])

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
        # Load the model
        model = xgb.XGBClassifier()
        model.load_model(str(save_model_path))
    else:
        print(f"Model does not exist at {save_model_path}. Training a new model.")

        # Train XGBoost model
        model = xgb.XGBClassifier(eval_metric='mlogloss', objective='multi:softmax', num_class=6, random_state=42)
        model.fit(X_train, y_train)

        # Save the model
        model.save_model(str(save_model_path))

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", conf_matrix)

    # Save reports
    with open("result/classification_report.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("result/confusion_matrix.png")