from models import random_forest, logistic_regression, gradient_boosting, knn, xgboost, X_train, y_train, X_val, y_val, X_test, y_test
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

RESULT_PATH= '../results'

def evaluate_model(model, name):

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_accuracy = accuracy_score(y_val, val_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    val_f1 = f1_score(y_val, val_pred, average="macro")
    test_f1 = f1_score(y_test, test_pred, average="macro")

    val_cm = confusion_matrix(y_val, val_pred)
    test_cm = confusion_matrix(y_test, test_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(val_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Home Win", "Draw", "Away Win"],
                yticklabels=["Home Win", "Draw", "Away Win"])

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    save_path = os.path.join(RESULT_PATH, f"confusion_matrix_{name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Print
    print("\n==============================")
    print(f"📊 Model: {name}")
    print("==============================")
    print(f"Validation Accuracy : {val_accuracy:.4f}")
    print(f"Validation F1 Score : {val_f1:.4f}")
    print("Validation Confusion Matrix:")
    print(val_cm)

    print("------------------------------")
    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Test F1 Score : {test_f1:.4f}")
    print("Test Confusion Matrix:")
    print(test_cm)
    print("==============================\n")

    return {
        "model": name,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "val_f1": val_f1,
        "test_f1": test_f1
    }

def compare_all_models():

    results = []

    results.append(
        evaluate_model(
            random_forest,
            name="Random Forest"
        )
    )

    results.append(
        evaluate_model(
            logistic_regression,
            name="Logistic Regression"
        )
    )

    results.append(
        evaluate_model(
            gradient_boosting,
            name="Gradient Boosting"
        )
    )

    results.append(
        evaluate_model(
            knn,
            name="KNN"
        )
    )

    results.append(
        evaluate_model(
            xgboost,
            name="XGBoost"
        )
    )

    results_df = pd.DataFrame(results)
    results_df= results_df.sort_values(by="val_accuracy", ascending=False)

    results_df.to_csv(os.path.join(RESULT_PATH, 'models_comparison.csv'), index=False)
    print(f'Model comparison was saved in {RESULT_PATH}')

    return results_df

def get_best_model(results_df):
    best_row = results_df.iloc[0]
    best_model_name = best_row['model']

    best_scores = {
        "val_accuracy": best_row["val_accuracy"],
        "test_accuracy": best_row["test_accuracy"],
        "val_f1": best_row["val_f1"],
        "test_f1": best_row["test_f1"]
    }

    print(f'🏆 Best Model: {best_model_name}')
    print("Scores associés :", best_scores)

    return best_model_name, best_scores

def get_rf_feature_importances():
    
    feat_importances = pd.Series(random_forest.feature_importances_, index=X_train.columns)
    feat_importances = feat_importances.sort_values(ascending=False)

    print("\n📊 Feature Importances Random Forest:")
    print(feat_importances)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=feat_importances.values, y=feat_importances.index, palette="viridis")
    plt.title("Feature Importances - Modèle Final")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.savefig(os.path.join(RESULT_PATH, 'rf_feature_importances.png'))

results= compare_all_models()
get_rf_feature_importances()
best_model_name, best_scores= get_best_model(results)
