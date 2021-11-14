import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv('./train.csv')

    df['kfold'] = -1
    
    y = df.target.values

    kf = model_selection.StratifiedKFold(n_splits = 4, shuffle = True)

    for f, (t_, v_) in enumerate(kf.split(X = df, y = y)):
        df.loc[v_, 'kfold'] = f

    df.to_csv('./train_folds.csv', index = False)