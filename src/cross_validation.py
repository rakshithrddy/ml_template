from sklearn import model_selection

"""
- -- binary_classification
- -- multiclass_classification
- -- multilabel_classification
- -- single_col_regression
- -- multi_col_regression
- -- holdout_
"""


class CrossValidation:
    def __init__(self,
                 df,
                 target_cols,
                 multilabel_delimiter=',',
                 problem_type="binary_classification",
                 num_folds=5,
                 random_state=42):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.multilabel_delimiter = multilabel_delimiter
        self.random_state = random_state
        self.dataframe['kfold'] = -1

    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value found in target column!")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds,
                                                     shuffle=False)
                for fold, (train_idx, val_idx) in enumerate(
                        kf.split(X=self.dataframe, y=self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type in ("single_col_regression", "multi_col_regression"):
            if self.num_targets != 1 and self.problem_type == 'single_col_regression':
                raise Exception("Invalid number of targets for this problem type")
            if self.num_targets < 2 and self.problem_type == 'multi_col_regression':
                raise Exception("Invalid number of targets for this problem type")
            kf = model_selection.KFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type.startswith("holdout_"):  # in case of a time series dataframe, and make shuffle as false
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, 'kfold'] = 0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, 'kfold'] = 1

        elif self.problem_type == "multilabel_classification":
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=targets)):
                self.dataframe.loc[val_idx, 'kfold'] = fold
        else:
            raise Exception(f"{self.problem_type} problem type not understood")
        return self.dataframe


