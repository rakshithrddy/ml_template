from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class FeatureExtractor:
    def __init__(self, X_train, X_validate, y_train, feature_extractor_type, n_components):
        self.X_train = X_train
        self.X_validate = X_validate
        self.y_train = y_train
        self.feature_extractor_type = feature_extractor_type
        self.n_components = n_components
        self.count = 0

    def pca(self):
        """
        info: pca is an unsupervised analysis, can be used only with linearly separable data
        """
        flag = True
        while flag:
            pca = PCA(n_components=self.n_components)
            self.X_train = pca.fit_transform(self.X_train)
            self.X_validate = pca.transform(self.X_validate)
            explainted_variance = pca.explained_variance_ratio_
            if self.n_components != len(explainted_variance):
                print(explainted_variance)
                required_variance = float(input("Enter the variance level to consider for training from the above table,"
                                          "value but be in the same format as the above data \n"))
                for variance in explainted_variance:
                    if float(variance) > required_variance:
                        self.count += 1
                self.n_components = self.count
                print(f'n_components used {self.n_components}')
                continue
            else:
                break

    def lda(self):
        """
        info: lda is an supervised analysis, can be used only with linearly separable data
        """
        lda = LDA(n_components=self.n_components)
        self.X_train = lda.fit_transform(self.X_train, self.y_train)  # in supervised we include y_train
        self.X_validate = lda.transform(self.X_validate)

    def kernalpca(self):
        """
        info: kernalpca is an supervised analysis, can be used with non linearly separable data
        """
        kpca = KernelPCA(n_components=self.n_components, kernel='rbf')
        self.X_train = kpca.fit_transform(self.X_train)  # in supervised we include y_train
        self.X_validate = kpca.transform(self.X_validate)


    def extact(self):
        print(f'performing feature extraction using {self.feature_extractor_type} method.')
        if self.feature_extractor_type == 'pca':
            self.pca()
            return self.X_train, self.X_validate, self.n_components
        elif self.feature_extractor_type == 'lda':
            self.lda()
            return self.X_train, self.X_validate, self.n_components
        elif self.feature_extractor_type == 'kpca':
            self.kernalpca()
            return self.X_train, self.X_validate, self.n_components
        else:
            raise Exception(f'feature extractor type {self.feature_extractor_type} not found')
