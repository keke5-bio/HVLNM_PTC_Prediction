# train_model.py
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import pickle
import os


def load_and_preprocess_data():
    """Load and preprocess data from the new directory."""
    print("Loading data...")

    # Update paths to match your new directory structure
    train_data_path = "data/训练集smote.csv"
    test_data_path = "data/验证集.csv"

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    # Updated feature list (11 predictors, ETE removed)
    features = [
        "New_focal", "CDFI", "SIRI_four", "HT", "TSH",
        "Tumor_size_custom", "NG", "Boundary", "Microcalcification",
        "LMR_four", "Special_location"
    ]
    target = "HVCLNM"

    # Ensure all features exist
    available_features = [f for f in features if f in train_data.columns and f in test_data.columns]
    print(f"Using features ({len(available_features)}): {available_features}")

    # Identify categorical features (CatBoost will handle them optimally)
    categorical_features = ["New_focal", "CDFI", "SIRI_four", "HT",
                            "Tumor_size_custom", "NG", "Boundary",
                            "Microcalcification", "LMR_four", "Special_location"]
    categorical_features = [f for f in categorical_features if f in available_features]
    print(f"Categorical features: {categorical_features}")

    # Extract features and target
    X_train = train_data[available_features]
    y_train = train_data[target]
    X_test = test_data[available_features]
    y_test = test_data[target]

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training set HVCLNM distribution:\n{y_train.value_counts(normalize=False)}")
    print(f"Test set HVCLNM distribution:\n{y_test.value_counts(normalize=False)}")

    return X_train, X_test, y_train, y_test, available_features, categorical_features


def train_catboost_model(X_train, X_test, y_train, y_test, features, cat_features):
    """Train the CatBoost model."""
    print("Training CatBoost model...")

    model = CatBoostClassifier(
        iterations=100,
        depth=5,
        learning_rate=0.03,
        random_seed=123,
        eval_metric='AUC',
        use_best_model=True,
        cat_features=cat_features,  # Explicitly specify categorical features
        verbose=100
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        plot=False,
        verbose=True
    )

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")

    # Confusion matrix metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision

    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision (PPV): {ppv:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy, auc, sensitivity, specificity, ppv


def save_model_and_features(model, features, cat_features, metrics_dict):
    """Save model and feature information."""
    os.makedirs('models', exist_ok=True)

    # Save the trained model
    with open('models/catboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Compile all information
    model_info = {
        'features': features,
        'categorical_features': cat_features,
        'metrics': metrics_dict,
        'feature_importance': dict(zip(features, model.get_feature_importance()))
    }

    with open('models/model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)

    print("Model and information saved successfully!")

    # Print sorted feature importance
    importance_sorted = sorted(model_info['feature_importance'].items(),
                               key=lambda x: x[1], reverse=True)
    print("\nFeature Importance Ranking:")
    for feature, importance in importance_sorted:
        print(f"  {feature}: {importance:.4f}")


def main():
    """Main execution function."""
    try:
        print(f"Current working directory: {os.getcwd()}")

        # 1. Load data
        X_train, X_test, y_train, y_test, features, cat_features = load_and_preprocess_data()

        # 2. Train model
        model = train_catboost_model(X_train, X_test, y_train, y_test, features, cat_features)

        # 3. Evaluate model
        accuracy, auc, sensitivity, specificity, ppv = evaluate_model(model, X_test, y_test)
        metrics_dict = {
            'accuracy': accuracy,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': ppv
        }

        # 4. Save model
        save_model_and_features(model, features, cat_features, metrics_dict)

        print(f"\nTraining completed! Final Model AUC: {auc:.4f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()