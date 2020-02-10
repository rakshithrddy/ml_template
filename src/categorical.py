from sklearn import preprocessing
import pandas as pd


class CategoricalFeatures:
    def __init__(self, dataframe, categorical_features, encoding_type, handle_na=False):
        """
        df: pandas dataframe
        categorical_features: list of column names, e.g. ["ord_1", "nom_0"......]
        encoding_type: label, binary, ohe
        handle_na: True/False
        """
        self.dataframe = dataframe
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        if self.handle_na:
            for c in self.cat_feats:
                self.dataframe.loc[:, c] = self.dataframe.loc[:, c].astype(str).fillna("-9999999")
        self.output_df = self.dataframe.copy(deep=True)

    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.dataframe[c].values.astype(str))
            self.output_df.loc[:, c] = lbl.transform(self.dataframe[c].values.astype(str))
            self.label_encoders[c] = lbl
        return self.output_df

    def _label_binarization(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.dataframe[c].values)
            val = lbl.transform(self.dataframe[c].values)
            self.output_df = self.output_df.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                self.output_df[new_col_name] = val[:, j]
            self.binary_encoders[c] = lbl
        return self.output_df

    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.dataframe[self.cat_feats].values)
        return ohe.transform(self.dataframe[self.cat_feats].values)

    def _dummy_encoder(self):
        for c in self.cat_feats:
            self.output_df.drop(c, inplace=True, axis=1)
            batch = pd.get_dummies(self.dataframe[c], prefix=c, drop_first=True)
            self.output_df = pd.concat([self.output_df, batch], axis=1)
        return self.output_df

    def fit_transform(self):  # transforms the whole data set
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarization()
        elif self.enc_type == "ohe":
            return self._one_hot()
        elif self.enc_type == 'dummy':
            return self._dummy_encoder()
        else:
            raise Exception("Encoding type not understood")

    def transform(self, dataframe):  # transform the particular given dataset
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna("-9999999")

        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe

        elif self.enc_type == "binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)

                for j in range(val.shape[1]):
                    new_col_name = c + f"__bin_{j}"
                    dataframe[new_col_name] = val[:, j]
            return dataframe

        elif self.enc_type == "ohe":
            return self.ohe(dataframe[self.cat_feats].values)
        else:
            raise Exception("Encoding type not understood")
