import joblib
import pandas as pd


class FatigueMLPredictor:
    def __init__(self, model_path, feature_columns_path, threshold_path):
        self.model = joblib.load(model_path)
        self.feature_columns = joblib.load(feature_columns_path)
        self.saved_threshold = float(joblib.load(threshold_path))

    def predict(self, feature_dict):
        x = pd.DataFrame([feature_dict])[self.feature_columns]
        prob_drowsy = float(self.model.predict_proba(x)[0][1])
        return prob_drowsy