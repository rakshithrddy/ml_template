from src import supporter
import pandas as pd
import math
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
import categorical
from cross_validation import CrossValidation
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer
from train import ModelTrainer
from feature_extractor import FeatureExtractor


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


class Main:
    def __init__(self,
                 train_csv,
                 test_csv,
                 submission_csv=None,
                 impute=False,
                 impute_method=None,
                 shuffle=False,
                 encoding=False,
                 encoder_attributes=None,
                 feature_scaling=True,
                 feature_scaling_type=None,
                 cross_validation=True,
                 cross_validation_attributes=None,
                 train_model=False,
                 train_model_attributes=None,
                 feature_extractor=False,
                 feature_extractor_attributes=None):
        """
        :param train_csv: takes train csv filename
        :param test_csv: takes test csv filename
        :param submission_csv: takes submission csv filename(not mandatory)
        :param impute: True/False
        :param impute_method: takes the method type to impute data
        (transformermix : Columns of dtype object are imputed with the most frequent value in column.
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
        self.transformermix_fit = DataFrameImputer()
        self.train_dataframe = pd.read_csv(self.folder_paths['path_to_train_csv'])
        self.test_dataframe = pd.read_csv(self.folder_paths['path_to_test_csv'])
        if submission_csv is not None:
            self.submission_dataframe = pd.read_csv(self.folder_paths['path_to_submission_csv'])

        self.original_train_dataframe = self.train_dataframe
        self.original_test_dataframe = self.test_dataframe
        self.impute = impute
        if self.impute:
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

        self.feature_extractor = feature_extractor
        if self.feature_extractor:
            self.feature_extractor_type = feature_extractor_attributes['extractor_type']
            if self.feature_extractor_type == 'pca':
                self.n_components = None
            elif self.feature_extractor_type == 'lda':
                n_features = len(self.original_train_dataframe.drop(feature_extractor_attributes['not_feature_cols'], axis=1).columns)
                unique_class = int(self.original_train_dataframe[self.target_column].nunique())
                self.n_components = int(min(n_features, (unique_class - 1)))
            else:
                self.n_components = 2

        self.X_train = None
        self.X_validate = None
        self.y_validate = None
        self.y_train = None



        self.FOLD_MAPPING = {
            0: [1, 2, 3, 4],
            1: [0, 2, 3, 4],
            2: [0, 1, 3, 4],
            3: [0, 1, 2, 4],
            4: [0, 1, 2, 3]
        }

    def data_imputer(self):
        self.train_dataframe.fillna(np.nan, inplace=True)
        self.test_dataframe.fillna(np.nan, inplace=True)
        if self.impute_method == 'transformermix':
            print(f"Imputing train and test dataframe using {self.impute_method}")
            transform_mix = DataFrameImputer()
            self.train_dataframe = transform_mix.fit_transform(self.train_dataframe)
            self.test_dataframe = transform_mix.fit_transform(self.test_dataframe)

        elif self.impute_method == "mean":
            print(f"Imputing train and test dataframe using {self.impute_method}")
            impute = SimpleImputer(missing_values=np.nan, strategy=self.impute_method)
            self.train_dataframe = impute.fit_transform(self.train_dataframe)
            self.test_dataframe = impute.transform(self.test_dataframe)

        elif self.impute_method == "median":
            print(f"Imputing train and test dataframe using {self.impute_method}")
            impute = SimpleImputer(missing_values=np.nan, strategy=self.impute_method)
            self.train_dataframe = impute.fit_transform(self.train_dataframe)
            self.test_dataframe = impute.transform(self.test_dataframe)

        elif self.impute_method == "most_frequent":
            print(f"Imputing train and test dataframe using {self.impute_method}")
            impute = SimpleImputer(missing_values=np.nan, strategy=self.impute_method)
            self.train_dataframe = impute.fit_transform(self.train_dataframe)
            self.test_dataframe = impute.transform(self.test_dataframe)

        elif self.impute_method == 'drop':
            print(f"Imputing train and test dataframe using {self.impute_method}")
            print(f"Total null value count in train dataframe= {self.train_dataframe.isnull().values.sum()}")
            print(f"Total null value count in test dataframe= {self.test_dataframe.isnull().values.sum()}")
            self.train_dataframe = self.train_dataframe.dropna(axis=0)
            self.test_dataframe = self.test_dataframe.dropna(axis=0)
        else:
            raise Exception(f"imputer method {self.impute_method} not available")

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
        if self.feature_scaling_type == 'minmax':
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
            raise Exception(f"feature scalar type {self.feature_extractor_type} not available ")
        return train_dataframe, test_dataframe

    def processer(self):
        if self.impute:
            self.data_imputer()

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
                self.train_dataframe, self.test_dataframe = self.categorical_encoder(
                    train_dataframe=self.train_dataframe,
                    test_dataframe=self.test_dataframe)
            else:
                raise Exception(f"{self.data_type} not available")

        if self.train_model:
            for fold in range(5):
                print(f"selecting fold {fold}")
                main_train = self.train_dataframe[self.train_dataframe.kfold.isin(self.FOLD_MAPPING.get(fold))]
                main_validate = self.train_dataframe[self.train_dataframe.kfold == fold]

                ########### splitting the train data frame into x_train, x_test, y_train, X_test ##############
                self.y_train = main_train[self.target_column].values
                self.y_validate = main_validate[self.target_column].values
                self.X_train = main_train.drop(["id", "target", "kfold"], axis=1)
                self.X_validate = main_validate.drop(["id", "target", "kfold"], axis=1)
                if self.feature_scaling:
                    print(f'feature scaling the dataset of fold {fold}')
                    self.X_train, self.X_validate = self.feature_scalar(train_dataframe=self.X_train,
                                                                        test_dataframe=self.X_validate)
                    if self.feature_extractor:
                        print(f"extracting features from the dataset of fold {fold} using {self.feature_extractor_type}")
                        feat_ext = FeatureExtractor(X_train=self.X_train, X_validate=self.X_validate,
                                                    feature_extractor_type=self.feature_extractor_type,
                                                    n_components=self.n_components, y_train=self.y_train)
                        self.X_train, self.X_validate, self.n_components = feat_ext.extact()


                train_instance = ModelTrainer(X_train=self.X_train,
                                              X_validate=self.X_validate,
                                              y_train=self.y_train,
                                              y_validate=self.y_validate,
                                              model_name=self.model_name)
                train_instance.train()


def starter():
    """
    - -- binary_classification  use only when the datasets' target column contains EXACTLY 2 types of classes
    - -- multiclass_classification use only when the datasets' target column contains more than 2 types of classes
    - -- multilabel_classification  use only when the dataset contains more than one target column. (delimiter: mandatory)
    - -- single_col_regression  use only when the dataset contains only on independent column(features) and one dependent column(labels)
    - -- multi_col_regression  use only when the dataset contains more than 1 independent column(features) and one depended column(label)
    - -- holdout_ use only when you want to split he dataset into a perticular train test ration.
                    use case: holdout_20 will splits the dataset into 80% train and 20% test
    - -- feature transform method: standard, minmax, maxabs, normalize
    - -- data imputer method: mean, median, most_frequent, transformermix, drop

    - --  model_names = linear_regression, randomforestclassifier, extratreesclassifier, polynomial_regression
    """
    encoder_attributes = {
        'non_categorical_features': ['id', 'bin_0', 'bin_1', 'bin_2', 'ord_0', 'day', 'month', 'kfold', 'target'],
        'encoder_type': 'label',
        'target': ['target'],
        'data_type': 'categorical'}

    cross_validation_attributes = {'multilabel_delimiter': "','",
                                   'problem_type': 'binary_classification',
                                   'num_folds': 5,
                                   'random_state': 42}
    train_model_attributes = {'model_name': 'logisticregression'}
    feature_extractor_attributes = {'extractor_type': 'pca',
                                    'not_feature_cols': ['target', 'id']}
    instance = Main(train_csv='train.csv',
                    test_csv='test.csv',
                    submission_csv=None,
                    impute=True,
                    impute_method='transformermix',
                    shuffle=True,
                    cross_validation=True,
                    cross_validation_attributes=cross_validation_attributes,
                    encoding=True,
                    encoder_attributes=encoder_attributes,
                    feature_scaling=True,
                    feature_scaling_type='minmax',
                    train_model=True,
                    train_model_attributes=train_model_attributes,
                    feature_extractor=True,
                    feature_extractor_attributes=feature_extractor_attributes
                    )
    instance.processer()


starter()
