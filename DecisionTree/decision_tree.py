import numpy as np
from copy import deepcopy


class DecisionTree:
    def __init__(self) -> None:
        self.tree = None

    def entropy(self, output_col):
        _, counts = np.unique(output_col, return_counts=True)
        entropy_val = -np.dot(counts/np.sum(counts),
                              np.log(counts/np.sum(counts)))
        return entropy_val

    def split_entropy(self, data, split_atribute, output):
        values, counts = np.unique(data[split_atribute], return_counts=True)
        entropy = [self.entropy(data.loc[data[split_atribute] ==
                   values[i]][output]) for i in range(len(values))]
        split_entropy_val = np.sum(counts/np.sum(counts) * entropy)
        return split_entropy_val

    def inf_gain(self, data, split_atribute, output):
        entropy = self.entropy(data[output])
        split_entropy = self.split_entropy(data, split_atribute, output)
        gain = entropy - split_entropy
        return gain

    def ID3(self, atributes, classes, data, best_leaf):
        if len(np.unique(data[classes])) <= 1:
            return np.unique(data[classes])[0]
        if len(atributes) == 0:
            return best_leaf
        class_values, num = np.unique(data[classes], return_counts=True)
        best_leaf = class_values[np.argmax(num)]
        gains = [self.inf_gain(data,
                               atribute,
                               classes) for atribute in atributes]
        best_atribute = atributes[np.argmax(gains)]
        tree = {best_atribute: {}}
        atributes.remove(best_atribute)
        for elem in np.unique(data[best_atribute]):
            updated_data = data.loc[data[best_atribute] == elem]
            nested_tree = self.ID3(deepcopy(atributes), classes,
                                   updated_data, best_leaf)
            tree[best_atribute][elem] = nested_tree
        self.tree = tree
        return tree

    def predict(self, data, atributes):
        tree = self.tree
        for _ in range(len(atributes)):
            for index, elem in enumerate(data):
                if atributes[index] in list(tree.keys()):
                    try:
                        result = tree[atributes[index]][elem]
                    except Exception:
                        random = np.random.choice(list(tree[
                            atributes[index]].keys()))
                        result = tree[atributes[index]][random]
                    if type(result) is str:
                        return result
                    else:
                        tree = result
                        continue
        return '?'
