from src import supporter
import pandas as pd
from cross_validation import CrossValidation
from train import ModelTrainer
from feature_extractor import FeatureExtractor
import data_preprocessing
from sklearn.model_selection import train_test_split
from utils import DataManipulate
from predict import Predicter
import joblib


# TODO fix one hot encoding with kfold crossValidation (still pending)


class Main:
    def __init__(self,
                 train_csv,
                 test_csv,
                 submission_csv=None,
                 impute_flag=False,
                 shuffle_flag=False,
                 encoding_flag=False,
                 encoder_attributes=None,
                 feature_scaling_flag=False,
                 cross_validation_flag=False,
                 train_model_flag=False,
                 feature_extractor_flag=False,
                 predict_flag=False,
                 attributes=None):
        """
        :param train_csv: takes train csv filename
        :param test_csv: takes test csv filename
        :param submission_csv: takes submission csv filename(not mandatory)
        :param impute_flag: bool
        :param shuffle_flag: bool
        :param encoding_flag: bool
        :param encoder_attributes: takes a string with the encoder_type followed by col names separated by |. 2 atbs are seperated by ||
        :param feature_extractor_flag: bool
        :param cross_validation_flag: bool
        :param train_model_flag: bool
        :param feature_scaling_flag: bool
        :param attributes: takes dictionary
        """

        ### import file names
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.submission_csv = submission_csv
        self.folder_paths = supporter.folder_paths(train_csv=self.train_csv,
                                                   test_csv=self.test_csv,
                                                   submission_csv=self.submission_csv)

        # read csv files with paths
        self.train_dataframe = pd.read_csv(self.folder_paths['path_to_train_csv'])
        self.test_dataframe = pd.read_csv(self.folder_paths['path_to_test_csv'])
        if submission_csv is not None:
            self.submission_dataframe = pd.read_csv(self.folder_paths['path_to_submission_csv'])
        self.path_to_models = self.folder_paths['path_to_models']
        # make a copy of dataframes
        self.original_train_dataframe = self.train_dataframe
        if self.test_csv is not None:
            self.original_test_dataframe = self.test_dataframe

        # initialize the required values
        self.impute_flag = impute_flag
        self.shuffle_flag = shuffle_flag
        self.encoding_flag = encoding_flag
        self.feature_scaling_flag = feature_scaling_flag
        self.cross_validation_flag = cross_validation_flag
        self.train_model_flag = train_model_flag
        self.feature_extractor_flag = feature_extractor_flag
        self.predict_flag = predict_flag
        self.attributes = attributes

        self.numerical_features = self.attributes['numerical_features']
        self.encoder_attributes = encoder_attributes
        self.target_columns = self.attributes['target_columns']
        self.index = self.attributes['index']
        self.data_type = self.attributes['data_type']
        self.cross_validation_type = self.attributes['cross_validation_type']
        self.multilabel_delimiter = self.attributes['multilabel_delimiter']
        self.problem_type = self.attributes['problem_type']
        self.n_kfold = self.attributes['n_kfold']
        self.random_state = self.attributes['random_state']
        self.model_name = self.attributes['model_name']
        self.feature_extractor_type = self.attributes['feature_extractor_type']
        self.data_impute_method = self.attributes['data_impute_method']
        self.feature_scaling_type = self.attributes['feature_scaling_type']
        self.submission_columns = self.attributes['submission_columns']
        self.classifier = None
        # initialize feature extractor values
        if self.feature_extractor_type == 'pca':
            self.n_components = None
        elif self.feature_extractor_type == 'lda':
            n_features = len(self.original_train_dataframe.drop(self.target_columns, axis=1).columns)
            unique_class = int(self.original_train_dataframe[self.target_columns].nunique())
            self.n_components = int(min(n_features - 1, (unique_class - 1)))
        else:
            self.n_components = 2
        self.data_preprocess_instance = data_preprocessing.PreProcessing(target_columns=self.target_columns,
                                                                         test_csv=test_csv,
                                                                         feature_scaling_type=self.feature_scaling_type,
                                                                         impute_method=self.data_impute_method)
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
        if self.impute_flag:
            instance = DataManipulate(train_dataframe=self.train_dataframe, target_columns=self.target_columns)
            self.train_dataframe = instance.scalar()
            self.train_dataframe, self.test_dataframe = self.data_preprocess_instance.data_imputer(
                train_dataframe=self.train_dataframe,
                test_dataframe=self.test_dataframe)
            # print(self.train_dataframe.isna().values.sum())
            # print(self.test_dataframe.isna().values.sum())
            # print(self.train_dataframe)
            # values = self.train_dataframe.value_counts().keys().tolist()
            # counts = self.train_dataframe.value_counts().tolist()
            # print(values)
            # print(counts)

        if self.shuffle_flag:
            print(f'Shuffling train and test dataframe')
            self.train_dataframe, self.test_dataframe = self.data_preprocess_instance.shuffle_data(
                train_dataframe=self.train_dataframe,
                test_dataframe=self.test_dataframe)

        if self.cross_validation_flag:
            print(f'Performing cross validation using {self.cross_validation_type} method.')
            if self.cross_validation_type == "kfold":
                print(f'cross validating the dataset using {self.problem_type} method')
                cross_instance = CrossValidation(dataframe=self.train_dataframe,
                                                 target_columns=self.target_columns,
                                                 multilabel_delimiter=self.multilabel_delimiter,
                                                 problem_type=self.problem_type,
                                                 n_kfold=self.n_kfold,
                                                 random_state=self.random_state)
                self.train_dataframe = cross_instance.split()
            elif self.cross_validation_type == 'train_test_split':
                X = self.train_dataframe.drop(self.target_columns, axis=1)
                y = self.train_dataframe[self.target_columns]
                self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(X, y, test_size=0.25,
                                                                                                random_state=None)

        if self.encoding_flag:
            list_of_encoder_attributes = self.encoder_attributes.split('||')
            for attribute in list_of_encoder_attributes:
                encoder_type, col = attribute.split("|")
                columns = col.split(',')
                print(f'Performing categorical encoding using {encoder_type}'
                      f' encoder for {self.cross_validation_type} cross validation.')
                if self.cross_validation_type == 'kfold':
                    self.train_dataframe, self.test_dataframe = self.data_preprocess_instance.kfold_categorical_encoder(
                        train_dataframe=self.train_dataframe,
                        test_dataframe=self.test_dataframe,
                        encoder_type=encoder_type,
                        feature_columns=columns)
                    print(f'After encoding, the total number of columns in train_dataframe = {len(self.train_dataframe.columns)}')
                    print(f'After encoding, the total number of columns in test_dataframe = {len(self.test_dataframe.columns)}')
                    # columns = ",".join(self.test_dataframe.columns)
                    # print(columns)
                elif self.cross_validation_type == 'train_test_split':
                    self.X_train, self.X_validate, self.test_dataframe = self.data_preprocess_instance.train_test_categorical_encoder(
                        X_train=self.X_train,
                        X_validate=self.X_validate,
                        test_dataframe=self.test_dataframe,
                        encoder_type=encoder_type,
                        feature_columns=columns)
                    print(f'After encoding, the total number of columns in X_train = {len(self.X_train.columns)}')
                    print(f'After encoding, the total number of columns in X_validate = {len(self.X_validate.columns)}')
                    print(f'After encoding, the total number of columns in test_dataframe = {len(self.test_dataframe.columns)}')
                    # columns = ",".join(self.test_dataframe.columns)
                    # print(columns)
                else:
                    raise Exception(f"{self.data_type} not available")

        if self.train_model_flag:
            if self.cross_validation_type == 'kfold':
                for fold in range(5):
                    print(f"selecting fold {fold}")
                    main_train = self.train_dataframe[self.train_dataframe.kfold.isin(self.FOLD_MAPPING.get(fold))]
                    main_validate = self.train_dataframe[self.train_dataframe.kfold == fold]
                    ########### splitting the train data frame into x_train, x_test, y_train, X_test ##############
                    self.y_train = main_train[self.target_columns].values
                    self.y_validate = main_validate[self.target_columns].values
                    self.X_train = main_train.drop([self.target_columns, self.index, "kfold"], axis=1)
                    self.X_validate = main_validate.drop([self.target_columns, self.index, "kfold"], axis=1)
                    if self.feature_scaling_flag:
                        print(f'feature scaling the dataset of fold {fold}')
                        self.X_train, self.X_validate = self.data_preprocess_instance.feature_scalar(
                            train_dataframe=self.X_train,
                            test_dataframe=self.X_validate)
                        if self.feature_extractor_flag:
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
                    self.classifier = train_instance.train()

            elif self.cross_validation_type == 'train_test_split':
                if self.feature_scaling_flag:
                    # self.X_train.drop(self.index, axis=1, inplace=True)
                    # self.X_validate.drop(self.index, axis=1, inplace=True)
                    print(f'feature scaling the dataset with {self.cross_validation_type} method.')
                    self.X_train, self.X_validate = self.data_preprocess_instance.feature_scalar(
                        train_dataframe=self.X_train,
                        test_dataframe=self.X_validate)
                    if self.feature_extractor_flag:
                        print(f"extracting features from the dataset using {self.feature_extractor_type}")
                        feat_ext = FeatureExtractor(X_train=self.X_train, X_validate=self.X_validate,
                                                    feature_extractor_type=self.feature_extractor_type,
                                                    n_components=self.n_components, y_train=self.y_train)
                        self.X_train, self.X_validate, self.n_components = feat_ext.extact()
                train_instance = ModelTrainer(X_train=self.X_train,
                                              X_validate=self.X_validate,
                                              y_train=self.y_train,
                                              y_validate=self.y_validate,
                                              model_name=self.model_name)
                self.classifier = train_instance.train()
        if self.predict_flag:
            # joblib.dump(self.classifier, f"{self.path_to_models}{self.model_name}.pkl")
            # joblib.dump(self.test_dataframe.columns, f"{self.path_to_models}{self.model_name}_columns.pkl")
            print('predicting the target values')
            predict_instance = Predicter(test_dataframe=self.test_dataframe,
                                         classifier=self.classifier,
                                         cross_validation_type=self.cross_validation_type,
                                         submission_columns=self.submission_columns,
                                         model_name=self.model_name,
                                         path_to_models=self.path_to_models,
                                         n_kfolds=self.n_kfold,
                                         attributes=self.attributes)
            predict_instance.predict()


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
    - -- encoder type : binary, label, ohe, dummy
    - -- cross_validation_type: train_test_split, kfold
    - -- feature extractor type: lda, pca, kpca
    - -- regressor_models : linear_regression, polynomial_regression, supportvector_regressor, decisiontree_regressor, kneighbors_regressor, randomforest_regressor
    - -- classifier_models : decisiontree_classifier, randomforest_classifier, extratrees_classifier, logistic_regression, kneighbors_classifier, supportvector_classifier
    - -- cluster_models : kmeans_cluster, hierarchical_cluster
    - -- "label|nom_0,nom_1,nom_2,nom_3,nom_4,nom_5,nom_6,nom_7,nom_8,nom_9||dummy|ord_1,ord_2,ord_3"
    """

    attributes = {
        # including index or id and target column in numerical_features
        'numerical_features': ['id', 'bin_0', 'bin_1', 'bin_2', 'ord_0' 'kfold', 'day', 'month', 'target'],
        'target_columns': ['target'],
        'index': 'id',
        'data_type': 'categorical',
        'cross_validation_type': 'train_test_split',
        'multilabel_delimiter': "','",
        'problem_type': 'binary_classification',
        'n_kfold': 5,
        'random_state': 42,
        'model_name': 'randomforest_classifier',
        'feature_extractor_type': 'lda',
        'data_impute_method': 'drop',
        'feature_scaling_type': 'minmax',
        'submission_columns': ['id', 'target']}
    encoder_attributes = "label|nom_0,nom_1,nom_2,nom_3,nom_4,nom_5,nom_6,nom_7,nom_8,nom_9||dummy|ord_0,ord_1,ord_2,ord_3,ord_4,ord_5,bin_3,bin_4"

    instance = Main(train_csv='train.csv',
                    test_csv='test.csv',
                    submission_csv=None,
                    impute_flag=True,
                    shuffle_flag=True,
                    cross_validation_flag=True,
                    encoding_flag=True,
                    encoder_attributes=encoder_attributes,
                    train_model_flag=True,
                    feature_scaling_flag=True,
                    feature_extractor_flag=False,
                    predict_flag=True,
                    attributes=attributes
                    )
    instance.processer()


starter()
