import pandas as pd
from sklearn.model_selection import train_test_split


cancer_df = pd.read_csv('breast-cancer.data',
                        names=['Class', 'age', 'menopause',
                               'tumor-size', 'inv-nodes',
                               'node-caps', 'deg-malig',
                               'breast', 'breast-quad',
                               'irradiat'], index_col=False)
mashroom_df = pd.read_fwf('agaricus-lepiota.data',
                          names=['Class', 'cap-shape', 'cap-surface',
                                 'cap-color', 'bruises', 'odor',
                                 'gill-attachment', 'gill-spacing',
                                 'gill-size', 'gill-color', 'stalk-shape',
                                 'stalk-root', 'stalk-surface-above-ring',
                                 'stalk-surface-below-ring',
                                 'stalk-color-above-ring',
                                 'stalk-color-below-ring',
                                 'veil-type', 'veil-color',
                                 'ring-number', 'ring-type',
                                 'spore-print-color',
                                 'population', 'habitat'],
                          delimiter=',', index_col=False)
cancer_Y = cancer_df.iloc[:, 0]
cancer_X = cancer_df.iloc[:, 1:]
mashroom_Y = mashroom_df.iloc[:, 0]
mashroom_X = mashroom_df.iloc[:, 1:]

X_train_cancer, X_test_cancer, \
    y_train_cancer, y_test_cancer = train_test_split(cancer_X,
                                                     cancer_Y,
                                                     test_size=0.4,
                                                     random_state=42)

X_train_mashroom, X_test_mashroom, \
    y_train_mashroom, y_test_mashroom = train_test_split(mashroom_X,
                                                         mashroom_Y,
                                                         test_size=0.4,
                                                         random_state=42)

DATA = {'cancer': (X_train_cancer, X_test_cancer,
                   y_train_cancer, y_test_cancer),
        'mashroom': (X_train_mashroom, X_test_mashroom,
                     y_train_mashroom, y_test_mashroom)}
