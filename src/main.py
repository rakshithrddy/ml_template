from src import supporter
import pandas as pd
import math
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
import categorical
from cross_validation import CrossValidation
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as sm


class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else math.ceil(X[c].mean()) for c in X],
            index=X.columns)
        return self

    def transform(self, X):
        return X.fillna(self.fill)


class Main:
    def __init__(self,
                 train_csv,
                 test_csv,
                 submission_csv=None,
                 impute_table=False,
                 impute_method=None,
                 shuffle=False,
                 encoding=False,
                 encoder_attributes=None,
                 feature_scaling=True,
                 feature_scaling_type=None,
                 cross_validation=True,
                 cross_validation_attributes=None,
                 train_model=False,
                 train_model_attributes=None):
        """
        :param train_csv: takes train csv filename
        :param test_csv: takes test csv filename
        :param submission_csv: takes submission csv filename(not mandatory)
        :param impute_table: True/False
        :param impute_method: takes the method type to impute data
        (TransformerMixin : Columns of dtype object are imputed with the most frequent value in column.
        Columns of other types are imputed with mean of column.)
        mean: fills the missing values with the mean of the column (applicable for numerical data only)
        medion: fills the missing values with the median of the column(applicable for numerical and classification)
        most_frequent: fills the missing values with the most frequent values (applicable for numerical and classification)
        :param shuffle: takes bool, if true, shuffles the dataset
        :param encoding: True/False
        :param encoder_attributes: dictionary of [encoding type, non_categorical_features, encoder_type, target]
        :param feature_scaling: True/False (if true normalize the dataset)
        """
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.submission_csv = submission_csv
        self.folder_paths = supporter.folder_paths(train_csv=self.train_csv,
                                                   test_csv=self.test_csv,
                                                   submission_csv=self.submission_csv)
        self.TransformerMixin_fit = DataFrameImputer()
        self.train_dataframe = pd.read_csv(self.folder_paths['path_to_train_csv'])
        self.test_dataframe = pd.read_csv(self.folder_paths['path_to_test_csv'])
        if submission_csv is not None:
            self.submission_dataframe = pd.read_csv(self.folder_paths['path_to_submission_csv'])

        self.impute_table = impute_table
        if self.impute_table:
            self.impute_method = impute_method
        self.shuffle = shuffle
        self.encoding = encoding
        self.data_type = encoder_attributes['data_type']
        if self.data_type == 'categorical':
            if encoder_attributes is None:
                raise Exception(f"For {self.data_type} method encoder attributes is mandatory")
            else:
                self.encoder_attributes = encoder_attributes
                self.non_categorical_features = encoder_attributes['non_categorical_features']
                self.encoding_type = encoder_attributes['encoder_type']
                self.target_column = encoder_attributes['target']
        self.feature_scaling = feature_scaling
        self.feature_scaling_type = feature_scaling_type
        self.cross_validation = cross_validation
        if self.cross_validation:
            self.multilabel_delimiter = cross_validation_attributes['multilabel_delimiter']
            self.problem_type = cross_validation_attributes['problem_type']
            self.num_folds = cross_validation_attributes['num_folds']
            self.random_state = cross_validation_attributes['random_state']

        self.train_model = train_model
        if self.train_model:
            self.train_attributes = train_model_attributes
            self.model_name = self.train_attributes['model_name']

    def data_imputer(self, dataframe):
        try:
            if self.impute_method == 'TransformerMixin' and self.data_type == 'numerical' or self.data_type == 'categorical':
                self.TransformerMixin_fit.fit(dataframe)
                return self.TransformerMixin_fit.transform(dataframe)
            elif self.impute_method == "mean" and self.data_type == 'numerical':
                impute = SimpleImputer(missing_values="NaN", strategy=self.impute_method)
                return impute.fit_transform(dataframe)
            elif self.impute_method == "median" and self.data_type == 'numerical' or self.data_type == 'categorical':
                impute = SimpleImputer(missing_values="NaN", strategy=self.impute_method)
                return impute.fit_transform(dataframe)
            elif self.impute_method == "most_frequent" and self.data_type == 'numerical' or self.data_type == 'categorical':
                impute = SimpleImputer(missing_values="NaN", strategy=self.impute_method)
                return impute.fit_transform(dataframe)
        except Exception:
            raise Exception(f"{self.impute_method} method is not suitable for {self.data_type}")

    @staticmethod
    def shuffle_data(dataframe):
        try:
            return dataframe.sample(frac=1).reset_index(drop=True)
        except Exception as e:
            raise e

    def categorical_encoder(self, train_dataframe, test_dataframe, handle_na=False):
        try:
            for target in self.target_column:
                test_dataframe[target] = -1
            full_dataframe = pd.concat([train_dataframe, test_dataframe])
            if self.encoding_type == 'ohe':
                train_dataframe_len = len(train_dataframe)
                categorical_feature_cols = [c for c in train_dataframe.columns if c not in self.non_categorical_features]
                category_encoder = categorical.CategoricalFeatures(df=full_dataframe,
                                                                   categorical_features=categorical_feature_cols,
                                                                   encoding_type=self.encoding_type,
                                                                   handle_na=handle_na)
                full_dataframe_encoded = category_encoder.fit_transform()
                train_dataframe_encoded = full_dataframe_encoded[:train_dataframe_len, :]
                test_dataframe_encoded = full_dataframe_encoded[train_dataframe_len:, :]
                return train_dataframe_encoded, test_dataframe_encoded
            elif self.encoding_type == 'label':
                train_idx = train_dataframe['id'].values
                test_idx = test_dataframe['id'].values
                categorical_feature_cols = [c for c in train_dataframe.columns if c not in self.non_categorical_features]
                category_encoder = categorical.CategoricalFeatures(df=full_dataframe,
                                                                   categorical_features=categorical_feature_cols,
                                                                   encoding_type=self.encoding_type,
                                                                   handle_na=handle_na)
                full_dataframe_encoded = category_encoder.fit_transform()
                train_dataframe_encoded = full_dataframe_encoded[full_dataframe_encoded['id'].isin(train_idx)].reset_index(drop=True)
                test_dataframe_encoded = full_dataframe_encoded[full_dataframe_encoded['id'].isin(test_idx)].reset_index(drop=True)
                return train_dataframe_encoded, test_dataframe_encoded
            elif self.encoding_type == 'binary':
                categorical_feature_cols = [c for c in train_dataframe.columns if c not in self.non_categorical_features]
                category_encoder = categorical.CategoricalFeatures(df=train_dataframe,
                                                                   categorical_features=categorical_feature_cols,
                                                                   encoding_type=self.encoding_type,
                                                                   handle_na=handle_na)
                train_dataframe_encoded = category_encoder.fit_transform()
                test_dataframe_encoded = category_encoder.transform(test_dataframe)
                return train_dataframe_encoded, test_dataframe_encoded
        except Exception as e:
            raise e


    def feature_scalar(self, train_dataframe, test_dataframe):
        # train_list = set(self.train_dataframe.columns.tolist())
        # test_list = set(self.test_dataframe.columns.tolist())
        # left_out_cols = list(train_list.difference(test_list))
        # for columns in left_out_cols:
        #     test_dataframe[columns] = -999999
        if self.feature_scaling_type == 'standard':
            scalar = StandardScaler()
            train_dataframe = scalar.fit_transform(train_dataframe)
            test_dataframe = scalar.transform(test_dataframe)
        if self.feature_scaling_type == 'minmax':
            scalar = MinMaxScaler()
            train_dataframe = scalar.fit_transform(train_dataframe)
            test_dataframe = scalar.transform(test_dataframe)
        elif self.feature_scaling_type == 'MaxAbs':
            scalar = MaxAbsScaler()
            train_dataframe = scalar.fit_transform(train_dataframe)
            test_dataframe = scalar.transform(test_dataframe)
        elif self.feature_scaling_type == 'normalize':
            scalar = Normalizer()
            train_dataframe = scalar.fit_transform(train_dataframe)
            test_dataframe = scalar.transform(test_dataframe)
        return train_dataframe, test_dataframe



    def processer(self):
        if self.impute_table:
            print(f"Imputing train and test dataframe using {self.impute_method}")
            self.train_dataframe = self.data_imputer(self.train_dataframe)
            self.test_dataframe = self.data_imputer(self.test_dataframe)

        if self.shuffle:
            print(f'Shuffling train and test dataframe')
            self.train_dataframe = self.shuffle_data(self.train_dataframe)
            self.test_dataframe = self.shuffle_data(self.test_dataframe)

        if self.cross_validation:
            print(f'cross validating the dataset using {self.problem_type} method')
            cross_instance = CrossValidation(df=self.train_dataframe,
                                             target_cols=self.target_column,
                                             multilabel_delimiter=self.multilabel_delimiter,
                                             problem_type=self.problem_type,
                                             num_folds=self.num_folds,
                                             random_state=self.random_state)
            self.train_dataframe = cross_instance.split()

        if self.encoding:
            if self.data_type == 'numerical':
                print(f'Performing categorical encoding using {self.encoding_type}')
                pass
            elif self.data_type == 'categorical':
                print(f'Performing categorical encoding using {self.encoding_type}')
                self.train_dataframe, self.test_dataframe = self.categorical_encoder(train_dataframe=self.train_dataframe,
                                                                                     test_dataframe=self.test_dataframe)
            else:
                raise Exception(f"{self.data_type} not available")

        # if self.feature_scaling:
        #     print('feature scaling the datasets')
        #     self.train_dataframe, self.test_dataframe = self.feature_scalar(train_dataframe=self.train_dataframe,
        #                                                                     test_dataframe=self.test_dataframe)

        if self.train_model:
            pass




def starter():
    """
    - -- binary_classification  use only when the datasets' target column contains EXACTLY 2 types of classes
    - -- multiclass_classification use only when the datasets' target column contains more than 2 types of classes
    - -- multilabel_classification  use only when the dataset contains more than one target column. (delimiter: mandatory)
    - -- single_col_regression  use only when the dataset contains only on independent column(features) and one dependent column(labels)
    - -- multi_col_regression  use only when the dataset contains more than 1 independent column(features) and one depended column(label)
    - -- holdout_ use only when you want to split he dataset into a perticular train test ration.
                    use case: holdout_20 will splits the dataset into 80% train and 20% test
    """
    encoder_attributes = {'non_categorical_features': ['id', 'bin_0', 'bin_1', 'bin_2', 'ord_0', 'day', 'month', 'kfold', 'target'],
                          'encoder_type': 'label',
                          'target': ['target'],
                          'data_type': 'categorical'}

    cross_validation_attributes = {'multilabel_delimiter': "','",
                                   'problem_type': 'binary_classification',
                                   'num_folds': 5,
                                   'random_state': 42}
    train_model_attributes = {'model_name':'linear_regression'}
    instance = Main(train_csv='train.csv',
                    test_csv='test.csv',
                    submission_csv=None,
                    impute_table=True,
                    impute_method='TransformerMixin',
                    shuffle=True,
                    cross_validation=True,
                    cross_validation_attributes=cross_validation_attributes,
                    encoding=True,
                    encoder_attributes=encoder_attributes,
                    feature_scaling=False,
                    feature_scaling_type='standard',
                    train_model=False,
                    train_model_attributes=train_model_attributes)
    instance.processer()

starter()