import joblib
from sklearn.preprocessing import PolynomialFeatures


class CommPredict:
    def __init__(self, num, comm_alo, comm_scale, ib="IB4") -> None:
        self.num = num / 1024
        self.comm_alo = comm_alo
        self.comm_scale = comm_scale
        self.ib = ib
        self.prediction = self._get_predict_res()

    def _get_predict_res(self):
        poly_features = PolynomialFeatures(degree=3, include_bias=False)
        X_pred = poly_features.fit_transform([[self.num]])
        model = joblib.load(f"./comm/{self.comm_alo}_{self.comm_scale}_{self.ib}_model.joblib")
        predictions = model.predict(X_pred) / 1000

        return predictions
