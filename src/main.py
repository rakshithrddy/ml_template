import sklearn
from src import supporter
import pandas as pd
import math
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
import cross_validation
import categorical
import cross_validation


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
                 fill_values=None,
                 shuffle=False,
                 data_type='numerical',
                 encoder_attributes=None,
                 feature_scaling=True):
        """
        :param train_csv: takes train csv filename
        :param test_csv: takes test csv filename
        :param submission_csv: takes submission csv filename(not mandatory)
        :param fill_values: takes the method type to impute data
        (TransformerMixin : Columns of dtype object are imputed with the most frequent value in column.
        Columns of other types are imputed with mean of column.)
        mean: fills the missing values with the mean of the column (applicable for numerical data only)
        medion: fills the missing values with the median of the column(applicable for numerical and classification)
        most_frequent: fills the missing values with the most frequent values (applicable for numerical and classification)
        :param shuffle: takes bool, if true, shuffles the dataset
        :param data_type: numerical, categorical,
        :param encoder_attributes: dictionary of [not_cat_feats, encoder_type, target]
        :param feature_scaling: True/False (if true normalize the dataset)
        """
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.submission_csv = submission_csv
        self.folder_paths = supporter.folder_paths(train_csv=self.train_csv,
                                                   test_csv=self.test_csv,
                                                   submission_csv=self.submission_csv)
        self.fill_values = fill_values
        self.TransformerMixin_fit = DataFrameImputer()
        self.shuffle = shuffle
        self.data_type = data_type

        # reading data files
        self.train_dataframe = pd.read_csv(self.folder_paths['path_to_train_csv'])
        self.test_dataframe = pd.read_csv(self.folder_paths['path_to_test_csv'])
        if submission_csv is not None:
            self.submission_dataframe = pd.read_csv(self.folder_paths['path_to_submission_csv'])

        if self.data_type == 'categorical':
            if encoder_attributes is None:
                raise Exception(f"For {self.data_type} method encoder attributes is mandatory")
            else:
                self.encoder_attributes = encoder_attributes
                self.not_categorical_features = encoder_attributes['not_cat_feats']
                self.encoding_type = encoder_attributes['encoder_type']
                self.target_column = encoder_attributes['target']
        self.feature_scaling = feature_scaling
        self.train_dataframe_encoded = None
        self.test_dataframe_encoded = None

    def data_imputer(self, dataframe):
        try:
            if self.fill_values == 'TransformerMixin' and self.data_type == 'numerical' or self.data_type == 'categorical':
                self.TransformerMixin_fit.fit(dataframe)
                return self.TransformerMixin_fit.transform(dataframe)
            elif self.fill_values == "mean" and self.data_type == 'numerical':
                impute = SimpleImputer(missing_values="NaN", strategy=self.fill_values)
                return impute.fit_transform(dataframe)
            elif self.fill_values == "median" and self.data_type == 'numerical' or self.data_type == 'categorical':
                impute = SimpleImputer(missing_values="NaN", strategy=self.fill_values)
                return impute.fit_transform(dataframe)
            elif self.fill_values == "most_frequent" and self.data_type == 'numerical' or self.data_type == 'categorical':
                impute = SimpleImputer(missing_values="NaN", strategy=self.fill_values)
                return impute.fit_transform(dataframe)
        except Exception:
            raise Exception(f"{self.fill_values} method is not suitable for {self.data_type}")

    @staticmethod
    def shuffle_data(dataframe):
        try:
            return dataframe.sample(frac=1).reset_index(drop=True)
        except Exception as e:
            raise e

    def categorical_encoder(self, train_dataframe, test_dataframe, handle_na=False):
        try:
            test_dataframe[self.target_column] = -1
            full_dataframe = pd.concat([train_dataframe, test_dataframe])
            if self.encoding_type == 'ohe':
                train_dataframe_len = len(train_dataframe)
                feature_cols = [c for c in train_dataframe.columns if c not in self.not_categorical_features]
                category_encoder = categorical.CategoricalFeatures(df=full_dataframe,
                                                                   categorical_features=feature_cols,
                                                                   encoding_type=self.encoding_type,
                                                                   handle_na=handle_na)
                full_dataframe_encoded = category_encoder.fit_transform()
                train_dataframe_encoded = full_dataframe_encoded[:train_dataframe_len, :]
                test_dataframe_encoded = full_dataframe_encoded[train_dataframe_len:, :]
                return train_dataframe_encoded, test_dataframe_encoded
            elif self.encoding_type == 'label':
                train_idx = train_dataframe['id'].values
                test_idx = test_dataframe['id'].values
                feature_cols = [c for c in train_dataframe.columns if c not in self.not_categorical_features]
                category_encoder = categorical.CategoricalFeatures(df=full_dataframe,
                                                                   categorical_features=feature_cols,
                                                                   encoding_type=self.encoding_type,
                                                                   handle_na=handle_na)
                full_dataframe_encoded = category_encoder.fit_transform()
                train_dataframe_encoded = full_dataframe_encoded[full_dataframe_encoded['id'].isin(train_idx)].reset_index(drop=True)
                test_dataframe_encoded = full_dataframe_encoded[full_dataframe_encoded['id'].isin(test_idx)].reset_index(drop=True)
                return train_dataframe_encoded, test_dataframe_encoded
            elif self.encoding_type == 'binary':
                feature_cols = [c for c in train_dataframe.columns if c not in self.not_categorical_features]
                category_encoder = categorical.CategoricalFeatures(df=train_dataframe,
                                                                   categorical_features=feature_cols,
                                                                   encoding_type=self.encoding_type,
                                                                   handle_na=handle_na)
                train_dataframe_encoded = category_encoder.fit_transform()
                test_dataframe_encoded = category_encoder.transform(test_dataframe)
                return train_dataframe_encoded, test_dataframe_encoded
        except Exception as e:
            raise e

    def dataset_cross_validation(self, train_dataset, test_dataset):
        pass


    def processer(self):
        if self.fill_values is not None:
            print(f"Imputing train and test dataframe using {self.fill_values}")
            self.train_dataframe = self.data_imputer(self.train_dataframe)
            self.test_dataframe = self.data_imputer(self.test_dataframe)

        if self.shuffle:
            print(f'Shuffling train and test dataframe')
            self.train_dataframe = self.shuffle_data(self.train_dataframe)
            self.test_dataframe = self.shuffle_data(self.test_dataframe)

        if self.data_type == 'numerical':
            print(self.data_type)
            pass
        elif self.data_type == 'categorical':
            print(self.data_type)
            self.train_dataframe_encoded, self.test_dataframe_encoded = self.categorical_encoder(train_dataframe=self.train_dataframe,
                                                                                                 test_dataframe=self.test_dataframe)
        else:
            raise Exception(f"{self.data_type} not available")

        # TODO feature scaling
        if self.feature_scaling:
            pass

        #TODO cross validation

        #TODO Model training


def starter():
    encoder_attributes = {'not_cat_feats': ['id', 'target'],
                          'encoder_type': 'label',
                          'target': 'target'}
    instance = Main(train_csv='train.csv',
                    test_csv='test.csv',
                    submission_csv=None,
                    fill_values='TransformerMixin',
                    shuffle=False,
                    data_type='categorical',
                    encoder_attributes=encoder_attributes,
                    feature_scaling=True)
    instance.processer()

starter()


