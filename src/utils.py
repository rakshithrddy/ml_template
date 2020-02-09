from sklearn.utils import resample
import pandas as pd


class DataManipulate:
    def __init__(self, train_dataframe, target_columns):
        self.train_dataframe = train_dataframe
        self.target_columns = target_columns
        self.target_column = [c for c in self.train_dataframe.columns if
                              c in self.target_columns]

    def up_scale(self, max_value, min_value, max_value_count, col_name):
        df_majority = self.train_dataframe[self.train_dataframe[col_name] == max_value]
        df_minority = self.train_dataframe[self.train_dataframe[col_name] == min_value]
        df_minority_upsampled = resample(df_minority,
                                         replace=True,  # sample with replacement
                                         n_samples=max_value_count,  # to match majority class
                                         random_state=123)  # reproducible results
        self.train_dataframe = pd.concat([df_majority, df_minority_upsampled])

    def down_scale(self, max_value, min_value, min_value_count, col_name):
        df_majority = self.train_dataframe[self.train_dataframe[col_name] == max_value]
        df_minority = self.train_dataframe[self.train_dataframe[col_name] == min_value]
        df_majority_downsampled = resample(df_majority,
                                           replace=True,
                                           n_samples=min_value_count,
                                           random_state=123)
        self.train_dataframe = pd.concat([df_majority_downsampled, df_minority])

    def scalar(self):
        for i, col_name in enumerate(self.target_column):
            values = self.train_dataframe[self.target_column[i]].value_counts().keys().tolist()
            counts = self.train_dataframe[self.target_column[i]].value_counts().tolist()
            max_count_index = counts.index(max(counts))
            max_value = values[max_count_index]
            max_value_count = counts[max_count_index]
            min_count_index = counts.index(min(counts))
            min_value = values[min_count_index]
            self.down_scale(max_value=max_value,
                            min_value=min_value,
                            min_value_count=max_value_count,
                            col_name=col_name)
        return self.train_dataframe
