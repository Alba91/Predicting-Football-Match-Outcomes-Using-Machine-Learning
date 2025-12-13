from .data_loader import football_df
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

TRAIN_SEASONS = ['2019-2020', '2020-2021', '2021-2022', '2022-2023']
VAL_SEASONS = ['2023-2024']
TEST_SEASONS = ['2024-2025']

X_train = football_df[football_df["Season"].isin(TRAIN_SEASONS)].drop(columns=["FTR", "Season"])
y_train = football_df[football_df["Season"].isin(TRAIN_SEASONS)]["FTR"]

X_val = football_df[football_df["Season"].isin(VAL_SEASONS)].drop(columns=["FTR", "Season"])
y_val = football_df[football_df["Season"].isin(VAL_SEASONS)]["FTR"]

X_test = football_df[football_df["Season"].isin(TEST_SEASONS)].drop(columns=["FTR", "Season"])
y_test = football_df[football_df["Season"].isin(TEST_SEASONS)]["FTR"]

def find_best_random_forest():

    # Paramètres à tester
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [6, 8, 12],
        "min_samples_leaf": [3, 5]
    }

    model = RandomForestClassifier(random_state=42)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X_train, y_train)

    print("\nBest parameters found:")
    print(grid.best_params_)

    # Évalue sur la validation
    val_pred = grid.best_estimator_.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print("Validation accuracy:", val_acc)

    return grid.best_params_

def final_random_forest_model():

    best_params = find_best_random_forest()

    final_model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=42
    )

    final_model.fit(X_train, y_train)

    return final_model

def final_logistic_regression_model():

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

def final_gradient_boosting_model():

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

def final_knn_model():

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    return model

def final_xgboost_model():

    model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

random_forest= final_random_forest_model()
logistic_regression= final_logistic_regression_model()
gradient_boosting= final_gradient_boosting_model()
knn= final_knn_model()
xgboost= final_xgboost_model()

print('Models were successfully initialized.')
