from sklearn import metrics
import dispatcher
import joblib
from metrics import Metrics


class ModelTrainer:
    def __init__(self, X_train, X_validate, y_train, y_validate, model_name):
        self.X_train = X_train
        self.X_validate = X_validate
        self.y_train = y_train
        self.y_validate = y_validate
        self.model_name = model_name

    def train(self):
        print(f'training with {self.model_name}')
        classifier = dispatcher.MODELS[self.model_name]
        classifier.fit(self.X_train, self.y_train.ravel())
        metric_instance = Metrics(classifier=classifier, X_validate=self.X_validate, y_validate=self.y_validate)
        metric_instance.get_model_performance()

        # joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
        # joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
        # joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")