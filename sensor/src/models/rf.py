from sklearn.ensemble import RandomForestClassifier

def build_rf_model(params: dict):
    """
    Builds and returns a Random Forest model.
    """
    return RandomForestClassifier(**params)
