from data_loader import football_df
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

TRAIN_SEASONS = ['2019-2020', '2020-2021', '2021-2022', '2022-2023']
VAL_SEASONS = ['2023-2024']
TEST_SEASONS = ['2024-2025']

## Random Forest Model

K_values = [20, 50, 100, 150]
home_adv_values = [0, 50, 100, 150]
n_estimators_list = [100, 200]
max_depth_list = [6, 8, 12]
min_samples_leaf_list = [3, 5]

results = []

for K in K_values:
    for home_adv in home_adv_values:

        X_train = football_df[football_df["Season"].isin(TRAIN_SEASONS)].drop(columns=["FTR", "Season"])
        y_train = football_df[football_df["Season"].isin(TRAIN_SEASONS)]["FTR"]

        X_val = football_df[football_df["Season"].isin(VAL_SEASONS)].drop(columns=["FTR", "Season"])
        y_val = football_df[football_df["Season"].isin(VAL_SEASONS)]["FTR"]
        
        X_test = football_df[football_df["Season"].isin(TEST_SEASONS)].drop(columns=["FTR", "Season"])
        y_test = football_df[football_df["Season"].isin(TEST_SEASONS)]["FTR"]

        for n in n_estimators_list:
            for depth in max_depth_list:
                for leaf in min_samples_leaf_list:

                    model = RandomForestClassifier(
                        n_estimators=n,
                        max_depth=depth,
                        min_samples_leaf=leaf,
                        random_state=42
                    )

                    model.fit(X_train, y_train)

                    val_pred = model.predict(X_val)
                    test_pred = model.predict(X_test)

                    val_acc = accuracy_score(y_val, val_pred)
                    test_acc = accuracy_score(y_test, test_pred)

                    results.append({
                        "K": K,
                        "home_adv": home_adv,
                        "n_estimators": n,
                        "max_depth": depth,
                        "min_samples_leaf": leaf,
                        "val_acc": val_acc,
                        "test_acc": test_acc
                    })

                    print("Testing Random Forest with "
                        f"K={K}, HA={home_adv} | n={n}, depth={depth}, leaf={leaf} "
                        f"=> Val={val_acc:.4f}, Test={test_acc:.4f}"
                    )

# Convertir en DataFrame pour analyser les meilleurs résultats
results_df = pd.DataFrame(results)

print("\nBest parameters")
print(results_df.sort_values(by="val_acc", ascending=False).head(10))
print('We will use the best of them')
