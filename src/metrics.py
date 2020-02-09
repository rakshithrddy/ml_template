from sklearn import metrics


class Metrics:
    def __init__(self, classifier, X_validate, y_validate):
        self.X_validate = X_validate
        self.classifier = classifier
        self.y_validate = y_validate


    def get_model_performance(self):
        prediction_probability = self.classifier.predict_proba(self.X_validate)[:, 1]
        prediction = self.classifier.predict(self.X_validate)
        roc_auc_score = metrics.roc_auc_score(self.y_validate, prediction_probability)
        confusion_matrix = metrics.confusion_matrix(self.y_validate, prediction)
        print("################# CONFUSION MATRIX ##########################\n\n", confusion_matrix)
        print(f"\n\nROC_AUC_SCORE = {roc_auc_score}")
