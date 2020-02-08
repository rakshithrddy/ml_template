from sklearn import ensemble
from sklearn import linear_model
from sklearn import preprocessing
"""
linear_regression, randomforestclassifier, extratreesclassifier, polynomial_regression
"""
MODELS = {
    "linear_regression": linear_model.LinearRegression(n_jobs=6),
    "logisticregression": linear_model.LogisticRegression(n_jobs=6),
    "randomforestclassifier": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=6, verbose=2),
    "extratreesclassifier": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=6, verbose=2),
    "polynomial_regression": preprocessing.PolynomialFeatures(degree=2),
}