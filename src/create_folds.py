import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")
    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)  # Return a random sample of items from an axis of object.
    print("Total null values are", df.isnull().values.sum())
    print(df.shape)
    print(df.isnull().values.sum() )
    if df.isnull().values.sum() > 0:
        # df.fillna(method='ffill', inplace=True)
        df.dropna(axis=0, inplace=True)
    print("Total null values after filling using (ffill)", df.isnull().values.sum())
    print(df.shape)
    print(df.isnull().values.sum())
    # kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=42)
    #
    # for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
    #     print(len(train_idx), len(val_idx))
    #     df.loc[val_idx, 'kfold'] = fold  # Access a group of rows and columns by label(s) or a boolean array.
    #
    # df.to_csv("../input/train_folds.csv", index=False)