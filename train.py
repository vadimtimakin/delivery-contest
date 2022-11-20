# Imports
import os
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import Pool, CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')

# Get the data data
df_src = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")
sub = pd.read_csv("dataset/sample_solution.csv")

N_SPLITS = 5  # N_SPLITS=1 provides training on the entire dataset

# Feature mapping:
# -1 - dropped
# 0 - not in use
# 1 - numerical feature
# 2 - categorical feature
# 3 - target-encoded feature
# 4 - stratify feature
# 5 - target
features = {
    "oper_type": 2,
    "oper_attr": 2,
    "type": 2, 
    "priority": 2,
    "class": 2,
    "mailtype": 2,
    "mailctg": 2,
    "directctg": 2,
    "postmark": 2,

    "word_1": 0,
    "word_2": 0,
    "word_3": 0,

    "index_oper": 1,
    "weight": 1,
    "transport_pay": 1,
    "weight_mfi": 1,
    "price_mfi": 1,
    "dist_qty_oper_login_1": 1,
    "total_qty_oper_login_1": 1,
    "total_qty_oper_login_0": 1,
    "total_qty_over_index_and_type": 1,
    "total_qty_over_index": 1,

    "is_privatecategory": 2,
    "is_in_yandex": 2,
    "is_return": 2,

    "is_wrong_sndr_name": 2,
    "is_wrong_rcpn_name": 0,
    "is_wrong_phone_number": 2,
    "is_wrong_address": 2,

    "stratify": 4,
    "label": 5,

    "name_mfi": -1,
    "mailrank": -1,
    "oper_type + oper_attr": -1,
}

# Fix the seed
SEED = 0xFACED
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)

for bag in range(100):
    df_negative = df_src[df_src["label"] == 0].sample(n=170980)
    df_positive = df_src[df_src["label"] == 1]
    df = pd.concat((df_negative, df_positive)).reset_index()

    # Prepare the data
    df["stratify"] = df["label"]
    for feature in [k for k, v in features.items() if v == 2]:
        df[feature] = df[feature].astype(str)
        test_df[feature] = test_df[feature].astype(str)

    Y = df[[k for k, v in features.items() if v == 5]]
    X = df[[k for k, v in features.items() if v in [1, 2, 3]]]
    test_features = test_df[[k for k, v in features.items() if v in [1, 2, 3]]]


    # Initialize the model
    cb_model = CatBoostRegressor(
                iterations=100,
                learning_rate=0.03,
                depth=12,

                l2_leaf_reg=1,
                grow_policy="Lossguide",
                max_leaves=1024,

                eval_metric='AUC',
                loss_function='CrossEntropy', 
                random_seed=SEED,
                thread_count=16,
                early_stopping_rounds=100,
                use_best_model=bool(True * (N_SPLITS != 1)),
                verbose=10,
                )

    if N_SPLITS != 1:
        # Build Cross-Validation
        df['fold'] = 0
        kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        stratify = df[[k for k, v in features.items() if v == 4]]
        for fold, (train_idx, val_idx) in enumerate(kfold.split(df, stratify)):
            df.loc[val_idx, 'fold'] = fold

        # Train block
        folds_scores = []
        test_preds = None
        oof_preds = None
        oof_labels = None

        for fold_num in range(N_SPLITS):
            print(f'Fold {fold_num + 1}')
            # Get the fold data
            train_index = df.loc[df['fold'] != fold_num].index
            val_index = df.loc[df['fold'] == fold_num].index
            train_data, eval_data = X.iloc[train_index], X.iloc[val_index]
            train_label, eval_label = Y.iloc[train_index], Y.iloc[val_index]
            test_features = test_df[[k for k, v in features.items() if v in [1, 2, 3]]]

            # Convert categorical features to numeric ones
            target = [k for k, v in features.items() if v == 5][0]
            modified_features = [k for k, v in features.items() if v == 3]
            for mf in modified_features:

                m = {}
                for u in train_data[mf].unique():
                    m[u] = train_label[train_data[mf] == u][target].mean()
                train_data[mf] = train_data[mf].map(m)
                eval_data[mf] = eval_data[mf].map(m)

                m = {}
                for u in test_features[mf].unique():
                    m[u] = Y[X[mf] == u][target].mean()
                test_features[mf] = test_features[mf].map(m)
            
            # Create datasets
            train_dataset = Pool(data=train_data,
                                label=train_label,
                                cat_features=[k for k, v in features.items() if v == 2],
                                )

            eval_dataset = Pool(data=eval_data,
                                label=eval_label,
                                cat_features=[k for k, v in features.items() if v == 2],
                                )

            # Fit the model
            cb_model.fit(train_dataset, eval_set=eval_dataset)
            cb_model.save_model(f'model_{fold_num}.cbm', format="cbm")

            # Do test and validation preds
            eval_cb_preds = cb_model.predict_proba(eval_data)[:, 1]
            test_cb_preds = cb_model.predict_proba(test_features)[:, 1]

            if test_preds is None:
                test_preds = test_cb_preds / N_SPLITS
                oof_preds = eval_cb_preds
                oof_labels = eval_label
            else:
                test_preds += test_cb_preds / N_SPLITS
                oof_preds = np.concatenate((oof_preds, eval_cb_preds))
                oof_labels = np.concatenate((oof_labels, eval_label))

            # Calculate and save scores
            folds_scores.append(roc_auc_score(eval_label, eval_cb_preds))

        # Print CV score and update log
        for fold, score in enumerate(folds_scores):
            print(f"Fold {fold + 1} : {round(score, 3)}")

        # Compute OOF score
        oof = pd.DataFrame({"preds": oof_preds, "labels": np.squeeze(oof_labels)})
        oof.to_csv("oof.csv", index=False)
        cv = round(sum(folds_scores) / len(folds_scores), 3)
        auc = round(roc_auc_score(oof_labels, oof_preds, multi_class='ovo'), 3)
        print("CV:", cv)
        print("OOF", auc)

        with open('log.txt', 'a') as file:
            file.write(f'CV: {cv} ')
            file.write(f'AUC: {auc} ')
            file.write('\n')

    else: 
        # Convert categorical features to numeric ones
        target = [k for k, v in features.items() if v == 5][0]
        modified_features = [k for k, v in features.items() if v == 3]
        for mf in modified_features:

            m = {}
            for u in X[mf].unique():
                m[u] = Y[X[mf] == u][target].mean()
            X[mf] = X[mf].map(m)

            m = {}
            for u in test_features[mf].unique():
                m[u] = Y[X[mf] == u][target].mean()
            test_features[mf] = test_features[mf].map(m)
        
        # Create dataset
        train_dataset = Pool(data=X,
                            label=Y,
                            cat_features=[k for k, v in features.items() if v == 2],
                            )

        # Fit the model
        cb_model.fit(train_dataset)
        train_preds = cb_model.predict_proba(X)[:, 1]
        test_preds = cb_model.predict_proba(test_features)[:, 1]

        # Compute score
        auc = round(roc_auc_score(Y, train_preds, multi_class='ovo'), 3)
        print("AUC", auc)

        with open('log.txt', 'a') as file:
            file.write(f'AUC: {auc} ')
            file.write('\n')

    # Create a submission 
    sub['label'] = test_preds 
    sub.to_csv(f'subs/sub_{bag}.csv', index=False)