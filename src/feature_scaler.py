from sklearn import preprocessing


class FeatureScaler:
    def __init__(self, X_train, X_test, scalar_type='standard'):
        self.X_Train = X_train
        self.X_Test = X_test
        self.scalar_type = scalar_type

    def standardized(self):
        sc = preprocessing.StandardScaler()
        self.X_Train = sc.fit_transform(self.X_Train)
        self.X_Test = sc.transform(self.X_Test)
        return self.X_Train, self.X_Test

    def normalizer(self):
        nm = preprocessing.Normalizer()


    def scale(self):
        if self.scalar_type == 'standard':
            return self.standardized()
        elif self.scalar_type == 'normal':
            pass
        else:
            raise Exception(f"feature scalar type {self.scalar_type} not found")