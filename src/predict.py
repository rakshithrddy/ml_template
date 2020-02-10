import pandas as pd
import numpy as np
import json
import joblib


class Predictor:
    def __init__(self, test_dataframe, cross_validation_type, submission_columns, model_name, attributes,
                 path_to_models, classifier):
        self.test_dataframe = test_dataframe
        self.cross_validation_type = cross_validation_type
        self.submission_columns = submission_columns
        self.model_name = model_name
        self.attributes = attributes
        self.path_to_models = path_to_models
        self.classifier = classifier

    def prediction(self):
        if self.cross_validation_type == 'train_test_split':
            test_idx = self.test_dataframe['id'].values
            prediction = self.classifier.predict(self.test_dataframe)
            submission = pd.DataFrame(np.column_stack((test_idx, prediction)), columns=self.submission_columns)
            submission.id = submission.id.astype(int)
            submission = submission.sort_values(['id'])
            if len(submission) < 400000:
                inp = input('the values are less than 400000, do you want to continue. y/n')
                if inp == 'y':
                    submission.to_csv(f"{self.path_to_models}{self.model_name}{self.cross_validation_type}.csv", index=False)
                else:
                    pass
            else:
                submission.to_csv(f"{self.path_to_models}{self.model_name}{self.cross_validation_type}.csv", index=False)
        else:
            raise Exception
        with open(f"{self.path_to_models}{self.model_name}{self.cross_validation_type}_attributes.json", 'w') as fp:
            json.dump(self.attributes, fp)
