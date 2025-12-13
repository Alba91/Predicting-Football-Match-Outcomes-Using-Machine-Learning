import src
from src.evaluation import evaluate_models

# To be used for a specific purpose if needed => Code below is not required for the project to work

football_data = src.football_df
random_forest_model = src.random_forest
logistic_regression_model = src.logistic_regression
gradient_boosting_model = src.gradient_boosting
knn_model = src.knn
xgboost_model = src.xgboost

# Specifically to evaluate as required

evaluate_models()
