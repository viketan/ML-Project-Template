import os
hyperparams = {
    "Linear Regression": {},  # No hyperparameters to tune
    "Ridge Regression": {
        'alpha': [0.1, 1.0, 10.0]
    },
    "Lasso Regression": {
        'alpha': [0.1, 1.0, 10.0]
    },
    "KNN Regression": {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    },
    "Decision Tree": {
        'max_depth': [3, 5, 7]
    },
    "Random Forest": {
        'n_estimators': [100, 500],
        'max_depth': [3, 5, 7]
    },
    "Adaboost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        'max_depth': [3, 5, 7]
    },
    "xgboost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        'max_depth': [3, 5, 7]
    },
    "LightBGM": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        'num_leaves': [31, 127]
    }
}
artifact_dir = "artifact"
PREPROCESSOR_FILEPATH = os.path.join(artifact_dir, "preprocessing.pkl")
MODEL_FILEPATH = os.path.join(artifact_dir, "model.pkl")