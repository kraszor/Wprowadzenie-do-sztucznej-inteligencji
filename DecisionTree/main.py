from data import DATA
from decision_tree import DecisionTree
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main():
    flag = True
    cancer_data = DATA['cancer']
    mashroom_data = DATA['mashroom']
    if flag:
        data = cancer_data
    else:
        data = mashroom_data
    training = pd.concat((data[0], data[2]), axis=1)
    tree = DecisionTree()
    tree.ID3(list(data[0].columns), data[2].name, training, None)
    equal = 0
    y_hat = []
    for i in range(len(data[1])):
        preditcion = tree.predict(data[1].iloc[i, :], data[1].columns)
        y_hat.append(preditcion)
        if data[3].iloc[i] == preditcion:
            equal += 1
    print(f'Accuracy: {round(equal/len(data[3]) * 100, 2)}')
    ConfusionMatrixDisplay.from_predictions(data[3], y_hat,
                                            xticks_rotation='horizontal')
    plt.show()


if __name__ == '__main__':
    main()
