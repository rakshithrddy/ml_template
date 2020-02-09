import pandas as pd
import numpy as np
import json


class Predicter:
    def __init__(self, test_dataframe, classifier, cross_validation_type, submission_columns, model_name, attributes,
                 path_to_models, n_kfolds):
        self.test_dataframe = test_dataframe
        print(self.test_dataframe.columns)
        self.classifier = classifier
        self.cross_validation_type = cross_validation_type
        self.prediction = None
        self.submission_columns = submission_columns
        self.model_name = model_name
        self.attributes = attributes
        self.path_to_models = path_to_models
        self.n_folds = n_kfolds

    def predict(self):
        if self.cross_validation_type == 'kfold':
            test_idx = self.test_dataframe['id'].values
            for FOLD in range(self.n_folds):
                print(FOLD)
                preds = self.classifier.predict_proba(self.test_dataframe)[:, 1]
                if FOLD == 0:
                    self.prediction = preds
                else:
                    self.prediction += preds
            self.prediction /= 5
            submission = pd.DataFrame(np.column_stack((test_idx, self.prediction)), columns=self.submission_columns)
            submission.id = submission.id.astype(int)
            submission.to_csv(f"{self.path_to_models}{self.model_name}.csv", index=False)
        elif self.cross_validation_type == 'train_test_split':
            test_idx = self.test_dataframe['id'].values
            self.prediction = self.classifier.predict_proba(self.test_dataframe)[:, 1]
            submission = pd.DataFrame(np.column_stack((test_idx, self.prediction)), columns=self.submission_columns)
            submission.id = submission.id.astype(int)
            submission.to_csv(f"{self.path_to_models}{self.model_name}.csv", index=False)
        else:
            raise Exception
        with open(f"{self.path_to_models}{self.model_name}_attributes.json", 'w') as fp:
            json.dump(self.attributes, fp)
