import pandas as pd
from sklearn.model_selection import train_test_split


def get_data():
    back_df = pd.read_csv("BayesianNetwork/data.csv", index_col=False)
    back_Y = back_df.iloc[:, -1]
    back_X = back_df.iloc[:, :-1]

    X_train_back, X_test_back, \
        y_train_back, y_test_back = train_test_split(back_X,
                                                     back_Y,
                                                     test_size=0.2,
                                                     random_state=42)

    DATA = {'back': (X_train_back, X_test_back,
                     y_train_back, y_test_back)}
    return DATA
