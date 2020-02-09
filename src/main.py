from src import supporter
import pandas as pd
from cross_validation import CrossValidation
from train import ModelTrainer
from feature_extractor import FeatureExtractor
import data_preprocessing
from metrics import Metrics


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
                n_features = len(self.original_train_dataframe.drop(feature_extractor_attributes['not_feature_cols'],
                                                                    axis=1).columns)
                unique_class = int(self.original_train_dataframe[self.target_column].nunique())
                self.n_components = int(min(n_features, (unique_class - 1)))
            else:
                self.n_components = 2

        self.data_preprocess_instance = data_preprocessing.PreProcessing(target_column=self.target_column,
                                                                         encoding_type=self.encoding_type,
                                                                         non_categorical_features=self.non_categorical_features,
                                                                         feature_scaling_type=self.feature_scaling_type,
                                                                         impute_method=self.impute_method)

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

    def processer(self):
        if self.impute:
            self.train_dataframe, self.test_dataframe = self.data_preprocess_instance.data_imputer(
                train_dataframe=self.train_dataframe,
                test_dataframe=self.test_dataframe)

        if self.shuffle:
            print(f'Shuffling train and test dataframe')
            self.train_dataframe, self.test_dataframe = self.data_preprocess_instance.shuffle_data(
                train_dataframe=self.train_dataframe,
                test_dataframe=self.test_dataframe)

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
                print(f'Performing categorical encoding using {self.encoding_type} encoder.')
                self.train_dataframe, self.test_dataframe = self.data_preprocess_instance.numerical_encoder(
                                                                        train_dataframe=self.train_dataframe,
                                                                        test_dataframe=self.test_dataframe)
            elif self.data_type == 'categorical':
                print(f'Performing categorical encoding using {self.encoding_type} encoder.')
                self.train_dataframe, self.test_dataframe = self.data_preprocess_instance.categorical_encoder(
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
                    self.X_train, self.X_validate = self.data_preprocess_instance.feature_scalar(
                        train_dataframe=self.X_train,
                        test_dataframe=self.X_validate)
                    if self.feature_extractor:
                        print(
                            f"extracting features from the dataset of fold {fold} using {self.feature_extractor_type}")
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
    train_model_attributes = {'model_name': 'xgb'}
    feature_extractor_attributes = {'extractor_type': 'lda',
                                    'not_feature_cols': ['target', 'id']}
    instance = Main(train_csv='train.csv',
                    test_csv='test.csv',
                    submission_csv=None,
                    impute=True,
                    impute_method='drop',
                    shuffle=True,
                    cross_validation=True,
                    cross_validation_attributes=cross_validation_attributes,
                    encoding=True,
                    encoder_attributes=encoder_attributes,
                    feature_scaling=True,
                    feature_scaling_type='standard',
                    feature_extractor=True,
                    feature_extractor_attributes=feature_extractor_attributes,
                    train_model=True,
                    train_model_attributes=train_model_attributes,
                    )
    instance.processer()


"""
regressor_models : linear_regression, polynomial_regression, supportvector_regressor, decisiontree_regressor, kneighbors_regressor, randomforest_regressor
classifier_models : decisiontree_classifier, randomforest_classifier, extratrees_classifier, logistic_regression, kneighbors_classifier, supportvector_classifier
cluster_models : kmeans_cluster, hierarchical_cluster
"""

starter()
