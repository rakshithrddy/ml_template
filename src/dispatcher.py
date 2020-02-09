from sklearn import ensemble
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn import cluster
"""
linear_regression, randomforestclassifier, extratreesclassifier, polynomial_regression
logisticregression performs both regression and classification
"""
MODELS = {
    "linear_regression": linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=6),
    "polynomial_regression": preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=True,
                                                              order='C'),
    "supportvector_regressor": svm.SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0,
                                       tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200,
                                       verbose=False, max_iter=-1),
    'decisiontree_regressor': tree.DecisionTreeRegressor(criterion='mse', splitter='best',
                                                         max_depth=None, min_samples_split=2,
                                                         min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                                         max_features=None, random_state=None,
                                                         max_leaf_nodes=None, min_impurity_decrease=0.0,
                                                         min_impurity_split=None, presort='deprecated',
                                                         ccp_alpha=0.0),
    'kneighbors_regressor': neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto',
                                                          leaf_size=30, p=2, metric='minkowski', metric_params=None,
                                                          n_jobs=6),
    'randomforest_regressor': ensemble.RandomForestRegressor(n_estimators=200, criterion='mse', max_depth=None,
                                                             min_samples_split=2, min_samples_leaf=1,
                                                             min_weight_fraction_leaf=0.0,
                                                             max_features='auto', max_leaf_nodes=None,
                                                             min_impurity_decrease=0.0,
                                                             min_impurity_split=None, bootstrap=True, oob_score=False,
                                                             n_jobs=None,
                                                             random_state=None, verbose=0, warm_start=False,
                                                             ccp_alpha=0.0,
                                                             max_samples=None),


    'decisiontree_classifier': tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,
                                                           min_samples_split=2,
                                                           min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                                           max_features=None,
                                                           random_state=None, max_leaf_nodes=None,
                                                           min_impurity_decrease=0.0,
                                                           min_impurity_split=None, class_weight=None,
                                                           presort='deprecated',
                                                           ccp_alpha=0.0),
    "randomforest_classifier": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=6, verbose=2),
    "extratrees_classifier": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=6, verbose=2),
    "logistic_regression": linear_model.LogisticRegression(n_jobs=6),
    'kneighbors_classifier': neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto',
                                                            leaf_size=30, p=2, metric='minkowski', metric_params=None,
                                                            n_jobs=6),
    'supportvector_classifier': svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True,
                                        probability=False, tol=0.001, cache_size=200, class_weight=None,
                                        verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False,
                                        random_state=None),


    'kmeans_cluster': cluster.KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                                     precompute_distances='auto', verbose=0, random_state=None,
                                     copy_x=True, n_jobs=None, algorithm='auto'),
    'hierarchical_cluster': cluster.AgglomerativeClustering(n_clusters=2, affinity='euclidean', memory=None,
                                                            connectivity=None, compute_full_tree='auto',
                                                            linkage='ward', distance_threshold=None),

}
