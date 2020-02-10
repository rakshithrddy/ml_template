from sklearn import metrics.
import numpy as np
# from sklearn.metrics import confusion_matrix, classification_report


class Metrics:
    def __init__(self, classifier, X_validate, y_validate):
        self.X_validate = X_validate
        self.classifier = classifier
        self.y_validate = y_validate


    def get_model_performance(self):
        try:
            prediction_probability = self.classifier.predict_proba(self.X_validate)[:, 1]
            prediction = self.classifier.predict(self.X_validate)
            roc_auc_score = metrics.roc_auc_score(self.y_validate, prediction_probability)
            confusion_matrix = metrics.confusion_matrix(self.y_validate, prediction)
            print(metrics.classification_report(self.y_validate, prediction))
            print("\n\n################# CONFUSION MATRIX ##########################\n\n", confusion_matrix)
            print(f"\n\nROC_AUC_SCORE = {roc_auc_score}\n\n")
            print('MAE (closer to 0 is better)', metrics.mean_absolute_error(self.y_validate, prediction))
            print('MSE (closer to 0 is better)', metrics.mean_squared_error(self.y_validate, prediction))
            print('RMSE (closer to 0 is better)', np.sqrt(metrics.mean_squared_error(self.y_validate, prediction)))
        except Exception as e:
            print(e)
