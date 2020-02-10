import pandas as pd
import numpy as np

class DataManipulate:
    def __init__(self, train_dataframe, target_columns, sampling_type='down_sample'):
        self.train_dataframe = train_dataframe
        self.target_columns = target_columns
        self.sampling_type = sampling_type
        self.df_by_unique_class = None
        self.max_value = None
        self.min_value = None
        self.max_class = None
        self.min_class = None

    def up_sampling(self):
        up_sampled_dataframe = self.df_by_unique_class[self.min_class].sample(self.max_value*0.7, replace=True, random_state=243)
        print(f'Total rows after up sampling the data set = {len(up_sampled_dataframe)}')
        return pd.concat([self.df_by_unique_class[self.max_class], up_sampled_dataframe], axis=0)

    def down_sampling(self):
        down_sampled_dataframe = self.df_by_unique_class[self.max_class].sample(self.min_value, replace=True, random_state=112)
        print(f'Total rows after Down sampling the data set = {len(down_sampled_dataframe)}')
        return pd.concat([down_sampled_dataframe, self.df_by_unique_class[self.min_class]], axis=0)

    # def downsample(self, freq, n):
    #     cumsums = []
    #     total = 0
    #     choices, weights = zip(*freq.items())
    #     for weight in weights:
    #         total += weight
    #         cumsums.append(total)
    #     assert 0 <= n <= total
    #     result = collections.Counter()
    #     for _ in range(n):
    #         rnd = random.uniform(0, total)
    #         i = bisect.bisect(cumsums, rnd)
    #         result[choices[i]] += 1
    #         cumsums = [c if idx < i else c - 1 for idx, c in enumerate(cumsums)]
    #         total -= 1
    #     return result

    def scale(self):
        dict_values = {}
        for target_column in self.target_columns:
            value_counts = self.train_dataframe[target_column].value_counts().tolist()
            unique_class = self.train_dataframe[target_column].value_counts().keys().tolist()
            self.df_by_unique_class = dict(tuple(self.train_dataframe.groupby(target_column)))
            for value in unique_class:
                dict_values[value] = value_counts[value]
            all_values = dict_values.values()
            self.max_class = max(dict_values, key=dict_values.get)
            self.min_class = min(dict_values, key=dict_values.get)
            self.max_value = max(all_values)
            self.min_value = min(all_values)
            if self.sampling_type == 'down_sample':
                print(f'performing down sampling of dataframe {self.sampling_type}')
                return self.down_sampling()
            elif self.sampling_type == 'up_sample':
                print(f'performing up sampling of dataframe using {self.sampling_type}')
                return self.up_sampling()
            else:
                raise Exception(f"sampling type {self.sampling_type} is not available")
