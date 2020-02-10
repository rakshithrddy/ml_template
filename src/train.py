import dispatcher
from metrics import Metrics
import joblib


class ModelTrainer:
    def __init__(self, X_train, X_validate, y_train, y_validate, test_dataframe,  model_name, cross_validation_type,
                 predict_flag=False, path_to_models=None):
        self.X_train = X_train
        self.X_validate = X_validate
        self.y_train = y_train
        self.y_validate = y_validate
        self.model_name = model_name
        self.cross_validation_type = cross_validation_type
        self.classifier = None
        self.predict_flag = predict_flag
        self.test_dataframe = test_dataframe
        self.path_to_models = path_to_models

    def train(self):
        if self.cross_validation_type == 'kfold':
            print(f'training with {self.model_name}')
            self.classifier = dispatcher.MODELS[self.model_name]
            try:
                self.classifier.fit(self.X_train, self.y_train.values.ravel())
            except Exception:
                self.classifier.fit(self.X_train, self.y_train.ravel())
            metric_instance = Metrics(classifier=self.classifier, X_validate=self.X_validate, y_validate=self.y_validate)
            metric_instance.get_model_performance()
            if self.predict_flag:
                pred_probability = self.classifier.predict_proba(self.test_dataframe)[:, 1]
                return pred_probability
            else:
                print("PREDICTOR NOT ACTIVE")
                pred_probability = None
                return pred_probability


        elif self.cross_validation_type == 'train_test_split':
            print(f'training with {self.model_name}')
            classifier = dispatcher.MODELS[self.model_name]
            try:
                classifier.fit(self.X_train, self.y_train.ravel())
            except:
                classifier.fit(self.X_train, self.y_train.values.ravel())
            metric_instance = Metrics(classifier=classifier, X_validate=self.X_validate, y_validate=self.y_validate)
            metric_instance.get_model_performance()
            joblib.dump(classifier, f"{self.path_to_models}{self.model_name}.pkl")
            return classifier