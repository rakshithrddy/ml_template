from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import math
import categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer


class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.

        Columns of dtypes object are imputed with the most frequent value
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


class PreProcessing:
    def __init__(self,
                 target_column,
                 encoding_type,
                 non_categorical_features,
                 feature_scaling_type,
                 impute_method='transformermix'):
        self.impute_method = impute_method
        self.target_column = target_column
        self.encoding_type = encoding_type
        self.non_categorical_features = non_categorical_features
        self.feature_scaling_type = feature_scaling_type

    def data_imputer(self, train_dataframe, test_dataframe):
        train_dataframe.fillna(np.nan, inplace=True)
        test_dataframe.fillna(np.nan, inplace=True)
        if self.impute_method == 'transformermix':
            print(f"Imputing train and test dataframe using {self.impute_method}")
            transform_mix = DataFrameImputer()
            train_dataframe = transform_mix.fit_transform(train_dataframe)
            test_dataframe = transform_mix.fit_transform(test_dataframe)

        elif self.impute_method == "mean":
            print(f"Imputing train and test dataframe using {self.impute_method}")
            impute = SimpleImputer(missing_values=np.nan, strategy=self.impute_method)
            train_dataframe = impute.fit_transform(train_dataframe)
            test_dataframe = impute.transform(test_dataframe)

        elif self.impute_method == "median":
            print(f"Imputing train and test dataframe using {self.impute_method}")
            impute = SimpleImputer(missing_values=np.nan, strategy=self.impute_method)
            train_dataframe = impute.fit_transform(train_dataframe)
            test_dataframe = impute.transform(test_dataframe)

        elif self.impute_method == "most_frequent":
            print(f"Imputing train and test dataframe using {self.impute_method}")
            impute = SimpleImputer(missing_values=np.nan, strategy=self.impute_method)
            train_dataframe = impute.fit_transform(train_dataframe)
            test_dataframe = impute.transform(test_dataframe)

        elif self.impute_method == 'drop':
            print(f"Imputing train and test dataframe using {self.impute_method}")
            print(f"Total null value count in train dataframe= {train_dataframe.isnull().values.sum()}")
            print(f"Total null value count in test dataframe= {test_dataframe.isnull().values.sum()}")
            train_dataframe = train_dataframe.dropna(axis=0)
            test_dataframe = test_dataframe.dropna(axis=0)
        else:
            raise Exception(f"imputer method {self.impute_method} not available")
        return train_dataframe, test_dataframe

    def shuffle_data(self, train_dataframe, test_dataframe):
        try:
            train_dataframe = train_dataframe.sample(frac=1).reset_index(drop=True)
            test_dataframe = test_dataframe.sample(frac=1).reset_index(drop=True)
            return train_dataframe, test_dataframe
        except Exception as e:
            raise e

    def categorical_encoder(self, train_dataframe, test_dataframe, handle_na=False):
        try:
            for target in self.target_column:
                test_dataframe[target] = -1
            full_dataframe = pd.concat([train_dataframe, test_dataframe])
            if self.encoding_type == 'ohe':
                train_dataframe_len = len(train_dataframe)
                categorical_feature_cols = [c for c in train_dataframe.columns if
                                            c not in self.non_categorical_features]
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
                categorical_feature_cols = [c for c in train_dataframe.columns if
                                            c not in self.non_categorical_features]
                category_encoder = categorical.CategoricalFeatures(df=full_dataframe,
                                                                   categorical_features=categorical_feature_cols,
                                                                   encoding_type=self.encoding_type,
                                                                   handle_na=handle_na)
                full_dataframe_encoded = category_encoder.fit_transform()
                train_dataframe_encoded = full_dataframe_encoded[
                    full_dataframe_encoded['id'].isin(train_idx)].reset_index(drop=True)
                test_dataframe_encoded = full_dataframe_encoded[
                    full_dataframe_encoded['id'].isin(test_idx)].reset_index(drop=True)
                return train_dataframe_encoded, test_dataframe_encoded
            elif self.encoding_type == 'binary':
                categorical_feature_cols = [c for c in train_dataframe.columns if
                                            c not in self.non_categorical_features]
                category_encoder = categorical.CategoricalFeatures(df=train_dataframe,
                                                                   categorical_features=categorical_feature_cols,
                                                                   encoding_type=self.encoding_type,
                                                                   handle_na=handle_na)
                train_dataframe_encoded = category_encoder.fit_transform()
                test_dataframe_encoded = category_encoder.transform(test_dataframe)
                return train_dataframe_encoded, test_dataframe_encoded
        except Exception as e:
            raise e

    def numerical_encoder(self, train_dataframe, test_dataframe):
        return train_dataframe, test_dataframe

    def feature_scalar(self, train_dataframe, test_dataframe):
        """
        :param train_dataframe: X_train dataframe
        :param test_dataframe: X_test dataframe
        :return: trasformed dataset in ndarray
        """
        if self.feature_scaling_type == 'standard':
            scalar = StandardScaler()
            train_dataframe = scalar.fit_transform(train_dataframe)
            test_dataframe = scalar.transform(test_dataframe)
        elif self.feature_scaling_type == 'minmax':
            scalar = MinMaxScaler()
            train_dataframe = scalar.fit_transform(train_dataframe)
            test_dataframe = scalar.transform(test_dataframe)
        elif self.feature_scaling_type == 'maxabs':
            scalar = MaxAbsScaler()
            train_dataframe = scalar.fit_transform(train_dataframe)
            test_dataframe = scalar.transform(test_dataframe)
        elif self.feature_scaling_type == 'normalize':
            scalar = Normalizer()
            train_dataframe = scalar.fit_transform(train_dataframe)
            test_dataframe = scalar.transform(test_dataframe)
        else:
            raise Exception(f"feature scalar type {self.feature_scaling_type} not available ")
        return train_dataframe, test_dataframe
