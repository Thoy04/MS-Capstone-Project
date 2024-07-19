import catboost as cb
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBRegressor

# reading in dataframe
df = pd.read_csv('C:/Users/sierr/Desktop/MSDS/Capstone/forest_df.csv')
# shuffling data
df = df.sample(frac=1).reset_index(drop=True)
# selecting columns to be used in training
train_cols = ['age', 'g', 'gs', 'mp_per_g', 'fg_per_g',
              'fg_pct', 'fg3_per_g', 'fg3_pct', 'fg2_per_g', 'fg2_pct', 'ft_per_g',
              'ft_pct', 'orb_per_g', 'drb_per_g', 'ast_per_g', 'stl_per_g',
              'blk_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g', 'mpg_diff',
              'ppg_diff', 'astpg_diff', 'rebpg_diff', 'stl_diff', 'blk_diff', 'tov_diff', ]
# selecting identifier columns
id_cols = ["season", "player", "team_id", "award_share", "won_mip", "is_synthetic"]
# identifying target column
target = 'award_share'

# creating df used for training
model_df = df[train_cols]
# creating id df for post training
train_id = df[id_cols]
# creating df of target vals
target_df = df[[target]]

# filtering the train df to allow for testing on data the model has not seen
train_df = model_df[train_id['season'] <= 2019]
train_targets = target_df[train_id['season'] <= 2019]
seasons = train_id.season.unique()
train_seasons = np.sort(seasons[seasons <= 2019])

test_df = model_df[train_id['season'] > 2019]
test_df2 = test_df[~train_id["is_synthetic"]]
test_seasons = seasons[seasons > 2019]
test_seasons = np.sort(test_seasons)
test_targets = target_df[(train_id['season'] > 2019) & (~train_id["is_synthetic"])]
test_id = train_id[(train_id["season"] > 2019) & (~train_id["is_synthetic"])]

warnings.filterwarnings("ignore", message="Boolean Series key will be reindexed")
np.random.seed(1)

# see models and params.txt for models and params

# HP Optimization
def objective(trial):
    counter = 0

    param = {
        'max_depth': trial.suggest_categorical('max_depth', [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]),
        'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 4]),
        'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10]),
        'n_estimators': trial.suggest_categorical('n_estimators',
                                                  [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]),
    }

    for season in train_seasons:
        print(f"Season: {season}")

        train_fold = train_df[train_id["season"] != season]
        train_target = train_targets[train_id["season"] != season]
        val_fold = train_df[(train_id["season"] == season) & (~train_id["is_synthetic"])]
        val_targets = train_targets[(train_id["season"] == season) & (~train_id["is_synthetic"])]
        val_id = train_id[(train_id["season"] == season) & (~train_id["is_synthetic"])]

        rfr = RandomForestRegressor(**param)
        rfr.fit(train_fold, train_target.to_numpy()[:, 0])

        preds = rfr.predict(val_fold)
        most_votes = np.argmax(preds)
        won_mip = val_id.iloc[[most_votes]]["won_mip"].values[0]
        if won_mip == 1:
            counter += 1
        else:
            counter = counter
    return counter


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

xgb_params = {'n_estimators': 121,
 'max_depth': 7,
 'learning_rate': 0.31481936178743364,
 'colsample_bytree': 0.5067762388773397,
 'min_child_weight': 18}

gbm_params = {'num_iterations': 110,
 'max_bin': 457,
 'max_depth': 8,
 'learning_rate': 0.4119792828422604,
 'num_leaves': 90}

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(model_df))

counter = 0
validation_scores = {"season": [], "mae": [], "mse": [], "won_mip": [], "was_top_two": [], "was_top_three": [],
                     "id_info": []}
# 1987 - 2019
for season in train_seasons:
    print(f"Season: {season}")

    train_fold = train_df[train_id["season"] != season]
    train_target = train_targets[train_id["season"] != season]
    val_fold = train_df[(train_id["season"] == season) & (~train_id["is_synthetic"])]
    val_targets = train_targets[(train_id["season"] == season) & (~train_id["is_synthetic"])]
    val_id = train_id[(train_id["season"] == season) & (~train_id["is_synthetic"])]

    xgb = XGBRegressor(**xgb_params)
    xgb.fit(train_fold, train_target.to_numpy()[:, 0])

    preds = xgb.predict(val_fold).flatten()

    mae = mean_absolute_error(preds, val_targets.to_numpy()[:, 0])
    mse = mean_squared_error(preds, val_targets.to_numpy()[:, 0])
    top_two = val_id.iloc[np.argsort(preds)[-2:]]
    was_top_two = sum(top_two["won_mip"]) > 0
    top_three = val_id.iloc[np.argsort(preds)[-3:]]
    was_top_three = sum(top_three["won_mip"]) > 0
    print("Predicted top three players in MVP voting with their actual award_share:")
    print(top_three.iloc[::-1])

    most_votes = np.argmax(preds)
    score = np.amax(preds)
    won_mip = val_id.iloc[[most_votes]]["won_mip"].values[0]
    if won_mip == 1:
        counter += 1
    else:
        counter = counter
    player = val_id.iloc[[most_votes]]["player"].values[0]

    validation_scores["season"].append(season)
    validation_scores["mae"].append(mae)
    validation_scores["mse"].append(mse)
    validation_scores["won_mip"].append(won_mip)
    validation_scores["was_top_two"].append(was_top_three)
    validation_scores["was_top_three"].append(was_top_three)
    validation_scores["id_info"].append(val_id.iloc[[most_votes]])

cbr_df = pd.DataFrame(validation_scores)
cbr_df['won_mip'].sum()
cbr_df['was_top_two'].sum()
cbr_df['was_top_three'].sum()
cbr_df['mae'].mean()
cbr_df['mse'].mean()


# xgb_df = pd.DataFrame(validation_scores)
# xgb_df['won_mip'].sum()
# xgb_df.groupby('won_mip')['mse'].mean()

# gbm_df = pd.DataFrame(validation_scores)
# gbm_df['won_mip'].sum()
# gbm_df.groupby('won_mip')['mse'].mean()



# gbm_params = study.best_params

counter = 0
validation_scores = {"season": [], "mae": [], "mse": [], "won_mip": [], "was_top_two": [], "was_top_three": [],
                     "id_info": []}
# 2020 - 2023
for season in test_seasons:
    print(f"Season: {season}")

    train_fold = model_df[train_id["season"] < season]
    train_target = target_df[train_id["season"] < season]
    val_fold = model_df[(train_id["season"] == season) & (~train_id["is_synthetic"])]
    val_targets = target_df[(train_id["season"] == season) & (~train_id["is_synthetic"])]
    val_id = train_id[(train_id["season"] == season) & (~train_id["is_synthetic"])]

    xgb = XGBRegressor(**xgb_params)
    gmb = lgb.LGBMRegressor(**gbm_params)
    cbr = cb.CatBoostRegressor()
    rfr = RandomForestRegressor()
    svr = make_pipeline(StandardScaler(), SVR())
    model = keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))

    xgb.fit(train_fold, train_target.to_numpy()[:, 0])
    gmb.fit(train_fold, train_target.to_numpy()[:, 0])
    cbr.fit(train_fold, train_target.to_numpy()[:, 0])
    rfr.fit(train_fold, train_target.to_numpy()[:, 0])
    svr.fit(train_fold, train_target.to_numpy()[:, 0])
    model.fit(train_fold, train_target.to_numpy()[:, 0])

    xgb_preds = xgb.predict(val_fold).flatten()
    gmb_preds = gmb.predict(val_fold).flatten()
    cbr_preds = cbr.predict(val_fold).flatten()
    rfr_preds = rfr.predict(val_fold).flatten()
    svr_preds = svr.predict(val_fold).flatten()
    nn_preds = model.predict(val_fold).flatten()

    preds = (
                xgb_preds * 0.225 + gmb_preds * 0.225 + cbr_preds * 0.225 + rfr_preds * .225 + svr_preds * .05 + nn_preds * .05)

    mae = mean_absolute_error(preds, val_targets.to_numpy()[:, 0])
    mse = mean_squared_error(preds, val_targets.to_numpy()[:, 0])
    top_two = val_id.iloc[np.argsort(preds)[-2:]]
    was_top_two = sum(top_two["won_mip"]) > 0
    top_three = val_id.iloc[np.argsort(preds)[-3:]]
    was_top_three = sum(top_three["won_mip"]) > 0
    print("Predicted top three players in MVP voting with their actual award_share:")
    print(top_three.iloc[::-1])

    most_votes = np.argmax(preds)
    score = np.amax(preds)
    won_mip = val_id.iloc[[most_votes]]["won_mip"].values[0]
    if won_mip == 1:
        counter += 1
    else:
        counter = counter
    player = val_id.iloc[[most_votes]]["player"].values[0]

    validation_scores["season"].append(season)
    validation_scores["mae"].append(mae)
    validation_scores["mse"].append(mse)
    validation_scores["won_mip"].append(won_mip)
    validation_scores["was_top_two"].append(was_top_three)
    validation_scores["was_top_three"].append(was_top_three)
    validation_scores["id_info"].append(val_id.iloc[[most_votes]])

cbr_df = pd.DataFrame(validation_scores)
cbr_df['won_mip'].sum()
cbr_df['was_top_two'].sum()
cbr_df['was_top_three'].sum()
cbr_df['mae'].mean()
cbr_df['mse'].mean()
