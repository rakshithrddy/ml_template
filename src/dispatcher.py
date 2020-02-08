from sklearn import ensemble
from sklearn import linear_model

MODELS = {
    "linear_regression": linear_model.LinearRegression(),
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
}