import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    model_input = pd.read_csv('model_input.csv')

    # Separate the rows where label is 5
    label_5_data = model_input[model_input['label'] == 5]
    # Randomly sample 10000 rows from label 5 data
    label_5_sampled = label_5_data.sample(n=10000, random_state=42)
    # Get the rows where label is not 5
    other_labels_data = model_input[model_input['label'] != 5]
    # Combine the sampled label 5 data with the other labels data
    model_input_filt = pd.concat([label_5_sampled, other_labels_data], ignore_index=True)

    # model_input_filt.to_csv('model_input_filt.csv', index=False)

    # Prepare features and labels
    X = model_input_filt.drop(columns=["subGraphID", "label"])
    y = model_input_filt["label"]

    # Encode label
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split into train/test sets (75% train / 25% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

    # Train XGBoost model
    model = xgb.XGBClassifier(eval_metric='mlogloss', objective='multi:softmax', num_class=6, random_state=42)
    model.fit(X_train, y_train)

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