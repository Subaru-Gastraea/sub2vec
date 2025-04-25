import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

import os
import sys

import argparse

# Add the parent directory to sys.path
# Executing path: ddx-on-ehr/models/sub2vec/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))     # ddx-on-ehr/models/
from XGB_Classifier import XGBoostClassifierWrapper

# Save model in "./save_model" directory
# Save classification report and confusion matrix in "./result" directory
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='base_xgboost_model.json', required=True, help='Name of the model file to save/load')
    parser.add_argument('--train_file', type=str, default='./train.csv', required=True, help='Path to the training data directory')
    parser.add_argument('--test_file', type=str, default='./test.csv', required=True, help='Path to the testing data directory')
    parser.add_argument('--aux_prob', action='store_true', default=False, help='Use auxiliary probabilities (graphgen similarities) for evaluation')
    parser.add_argument('--prob_ratio', type=float, default=1.0, help='Weight for the main model probabilities')
    parser.add_argument('--aux_prob_ratio', type=float, default=1.0, help='Weight for the auxiliary probabilities')
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)
    num_classes  = 6

    # Prepare features and labels
    X_train = train_df.drop(columns=["subGraphID", "label"])
    X_test = test_df.drop(columns=["subGraphID", "label"])
    y_train = train_df["label"]
    y_test = test_df["label"]

    model_path = pathlib.Path("./save_model")
    model_path.mkdir(parents=True, exist_ok=True)
    save_model_path = model_path / args.model_name
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
    if args.aux_prob:
        # Load auxiliary probabilities
        aux_prob = pd.read_csv("../graphgen/all_norm_avg_sims.csv", header=None)
        aux_prob = aux_prob.to_numpy()
        report, conf_matrix = model.evaluate_proba(X_test, y_test, aux_proba=aux_prob, prob_ratio=args.prob_ratio, aux_prob_ratio=args.aux_prob_ratio)
    else:
        # Evaluate without auxiliary probabilities
        report, conf_matrix = model.evaluate(X_test, y_test)

    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", conf_matrix)

    # Save reports
    pathlib.Path("result").mkdir(parents=True, exist_ok=True)
    with open("result/classification_report.txt", "w") as f:
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