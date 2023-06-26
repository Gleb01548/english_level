import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
import catboost
import optuna

RANDOM_SEED = 121


df = pd.read_csv("data/interim/df_text.csv")
df = df[["text", "level"]]

train_df, test_df = train_test_split(df, train_size=0.8, random_state=0)
y_train, X_train = train_df["level"], train_df.drop(["level"], axis=1)
y_test, X_test = test_df["level"], test_df.drop(["level"], axis=1)

train_pool = Pool(data=X_train, label=y_train, text_features=["text"])
test_pool = Pool(data=X_test, label=y_test, text_features=["text"])

best_iter, best_score = None, float("-inf")


def objective(trial):
    global best_iter, best_score

    param = {
        "iterations": 10000,
        "early_stopping_rounds": 300,
        "task_type": "GPU",
        "devices": "0",
        "eval_metric": "TotalF1",
        "loss_function": "MultiClass",
        "random_seed": RANDOM_SEED,
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 2, 50),
        # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.8),
        "auto_class_weights": trial.suggest_categorical(
            "auto_class_weights", ["SqrtBalanced", "Balanced", "None"]
        ),
        "depth": trial.suggest_int("depth", 3, 9),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli"]
        ),
        "eval_metric": "TotalF1",
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 20)

    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    stat = cv(
        pool=train_pool,
        params=param,
        stratified=True,
        early_stopping_rounds=300,
        fold_count=5,
        logging_level="Silent",
    )
    score = stat["test-TotalF1-mean"].max()
    iterations = np.argmax(stat["test-TotalF1-mean"])
    if score > best_score:
        best_score = score
        best_iter = iterations
    return score


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
