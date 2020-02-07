from sklearn.linear_model import LinearRegression


class RegressionModels:
    def __init__(self, X_train, y_train, X_test, y_test, normalize=False, n_jobs=2):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.normalize = normalize
        self.n_jobs = n_jobs

    def linear_regressor(self):
        regressor = LinearRegression(normalize=self.normalize, n_jobs=self.n_jobs)
        regressor.fit(self.X_train, self.y_train)
        return regressor

    # def multi_liner_regressor(self):
    #     multi_regressor =