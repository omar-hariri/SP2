from xgboost import XGBClassifier

def build_xgb_model(params: dict):
    """
    Builds and returns an XGBoost model.
    """
    return XGBClassifier(**params)
