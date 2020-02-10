import matplotlib.pyplot as plt
import pandas as np
import os
from pandas.plotting import scatter_matrix
import supporter
import pandas as pd


class Visualize:
    def __init__(self, train_dataframe, path_to_visualisation, test_dataframe=None):
        self.train_dataframe = train_dataframe
        self.test_dataframe = test_dataframe
        self.path_to_visualisation = path_to_visualisation
        self.train_flag = True
        if not os.path.exists(self.path_to_visualisation):
            os.mkdir(self.path_to_visualisation)

    def histograms(self):
        print('creating histograms')
        if self.train_flag:
            self.train_dataframe.hist()
            plt.savefig(f'{self.path_to_visualisation}histograms_train.png')
        else:
            self.test_dataframe.hist()
            plt.savefig(f'{self.path_to_visualisation}histograms_test.png')
        return


    def density(self):
        print('creating density chart')
        if self.train_flag:
            self.train_dataframe.plot(kind='density', subplots=True, sharex=False)
            plt.savefig(f'{self.path_to_visualisation}density_train.png')
        else:
            self.test_dataframe.plot(kind='density', subplots=True, sharex=False)
            plt.savefig(f'{self.path_to_visualisation}density_test.png')
        return

    def box(self):
        print('creating box plot')
        if self.train_flag:
            self.train_dataframe.plot(kind='box', subplots=True, sharex=False, sharey=False)
            plt.savefig(f'{self.path_to_visualisation}box_train.png')
        else:
            self.test_dataframe.plot(kind='box', subplots=True, sharex=False, sharey=False)
            plt.savefig(f'{self.path_to_visualisation}box_test.png')
        return

    def correlation_matrix(self):
        print('creating correlation matrix')
        if self.train_flag:
            correlations = self.train_dataframe.corr()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(correlations, vmin=-1, vmax=1)
            fig.colorbar(cax)
            ticks = np.arange(0, 9, 1)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(self.train_dataframe.columns)
            ax.set_yticklabels(self.train_dataframe.columns)
            plt.savefig(f'{self.path_to_visualisation}correlation_matrix_train.png')
        else:
            correlations = self.test_dataframe.corr()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(correlations, vmin=-1, vmax=1)
            fig.colorbar(cax)
            # ticks = np.ar(0, 9, 1)
            # ax.set_xticks(ticks)
            # ax.set_yticks(ticks)
            ax.set_xticklabels(self.test_dataframe.columns)
            ax.set_yticklabels(self.test_dataframe.columns)
            plt.savefig(f'{self.path_to_visualisation}correlation_matrixs_test.png')
        return

    def scatterplot_matrix(self):
        if self.train_flag:
            scatter_matrix(self.train_dataframe)
            plt.savefig(f'{self.path_to_visualisation}scatterplot_matrix_train.png')
        else:
            scatter_matrix(self.test_dataframe)
            plt.savefig(f'{self.path_to_visualisation}scatterplot_matrix_test.png')
        return

    def sort_missing_values(self):
        if self.train_flag:
            print(self.train_dataframe.info())
            percent_missing = self.train_dataframe.isnull().sum() / self.train_dataframe.shape[0] * 100.00
            null_columns = self.train_dataframe.columns[self.train_dataframe.isnull().any()]
            counts = self.train_dataframe[null_columns].isnull().sum()
            print('missing values in percentage: \n\n', percent_missing)
            print('\n\nmissing values count: \n', counts)
            print(self.train_dataframe.describe())
        else:
            print(self.test_dataframe.info())
            percent_missing = self.test_dataframe.isnull().sum() / self.test_dataframe.shape[0] * 100.00
            null_columns = self.test_dataframe.columns[self.test_dataframe.isnull().any()]
            counts = self.test_dataframe[null_columns].isnull().sum()
            print('missing values in percentage: \n\n', percent_missing)
            print('\n\nmissing values count: \n', counts)
            print(self.test_dataframe.describe())
        return

    def visualize_data(self):
        # self.histograms()
        # self.density()
        # self.box()
        # self.correlation_matrix()
        # self.scatterplot_matrix()
        self.sort_missing_values()


# def instance():
#     pd.set_option('display.max_columns', 25)
#     folder_paths = supporter.folder_paths(train_csv='train.csv', test_csv='test.csv',submission_csv='submission.csv')
#     train_dataframe = np.read_csv(folder_paths['path_to_train_csv'])
#     test_dataframe = np.read_csv(folder_paths['path_to_test_csv'])
#     path_to_visualisation = folder_paths['path_to_visualisation']
#     visual_instance = Visualize(train_dataframe, path_to_visualisation, test_dataframe=test_dataframe)
#     visual_instance.visualize_data()

# instance()
