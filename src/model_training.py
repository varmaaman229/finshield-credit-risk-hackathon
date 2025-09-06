import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

def train_xgboost(X_train, y_train):
    """Trains an XGBoost model."""
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train):
    """Trains a LightGBM model."""
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

if __name__ == '__main__':
    # Load the processed data
    train_df = pd.read_csv('data/processed/train.csv')

    # Define features and target
    features = [col for col in train_df.columns if col not in ['SK_ID_CURR', 'TARGET']]
    X = train_df[features]
    y = train_df['TARGET']

    # Split the data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the models
    xgb_model = train_xgboost(X_train, y_train)
    lgb_model = train_lightgbm(X_train, y_train)

    # Evaluate the models
    xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
    lgb_preds = lgb_model.predict_proba(X_val)[:, 1]

    print(f"XGBoost AUC: {roc_auc_score(y_val, xgb_preds)}")
    print(f"LightGBM AUC: {roc_auc_score(y_val, lgb_preds)}")

    # Save the models
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    joblib.dump(lgb_model, 'models/lightgbm_model.pkl')

    print("Model training complete.")
