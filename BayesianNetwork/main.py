from network import BayesianNetwork
from decision_tree import DecisionTree
import pandas as pd
from data import get_data
# from sklearn.metrics import ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
import numpy as np


NETWORK = "BayesianNetwork/back_network.json"
TABLES = "BayesianNetwork/probability_tables.csv"


def main() -> None:
    for num in (50, 100, 200, 300, 500, 1000, 5000):
        test_acc = []
        for _ in range(25):
            network_obj = BayesianNetwork(num)
            network_obj.read_tables(TABLES)
            network_obj.read_network(NETWORK)
            df = network_obj.generate_data()
            network_obj.data_to_csv(df)
            DATA = get_data()
            back_data = DATA['back']
            training = pd.concat((back_data[0], back_data[2]), axis=1)
            tree = DecisionTree()
            tree.ID3(list(back_data[0].columns), back_data[2].name,
                     training, None)
            equal = 0
            y_hat = []
            for i in range(len(back_data[1])):
                preditcion = tree.predict(back_data[1].iloc[i, :],
                                          back_data[1].columns)
                y_hat.append(preditcion)
                if back_data[3].iloc[i] == preditcion:
                    equal += 1
            accuracy = round(equal/len(back_data[3]) * 100, 2)
            test_acc.append(accuracy)
            # ConfusionMatrixDisplay.from_predictions(back_data[3], y_hat,
            #                                         xticks_rotation='horizontal')
            # plt.show()
            # cm.figure_.savefig(f'cm/test_{self.dataset}_{model}_cm.png')
        print(f"\nTest stats after 25 attempts with data length: {num}")
        mean = np.mean(test_acc)
        std = np.std(test_acc)
        max_ = max(test_acc)
        min_ = min(test_acc)
        print(f"Mean: {mean}    std: {std}  max: {max_}  min: {min_}")


if __name__ == '__main__':
    main()
