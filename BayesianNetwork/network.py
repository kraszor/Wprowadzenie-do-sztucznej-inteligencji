import numpy as np
import pandas as pd
import itertools as it
from io import StringIO
import json


class BayesianNetwork:
    def __init__(self, data_amount) -> None:
        self.data_amount = data_amount
        self.network = None
        self.prob_tables = {}

    def read_network(self, network) -> None:
        with open(f'./{network}') as file_handle:
            self.network = json.load(file_handle)

    def read_tables(self, file) -> None:
        tables = {}
        joiner = "".join
        with open(f'./{file}') as file_handle:
            for key, group in it.groupby(file_handle,
                                         lambda line: line.startswith('&')):
                if not key:
                    group = tuple(group)
                    tables[group[0].rstrip()] = "".join(map(joiner, group[1:]))
        for elem in tuple(tables.keys()):
            a = StringIO(tables[elem])
            df = pd.read_csv(a, sep=",")
            self.prob_tables[elem] = df

    def get_network_keys(self, network, network_keys=[]) -> list:
        if isinstance(network, dict):
            network_keys += network.keys()
            v = network.values()
            for x in v:
                self.get_network_keys(x, network_keys)
        return network_keys

    def generate_data(self) -> pd.DataFrame:
        sequence = self.get_network_keys(self.network, [])[::-1]
        df = pd.DataFrame(columns=sequence)
        for i in range(self.data_amount):
            for elem in sequence:
                table = self.prob_tables[elem]
                try:
                    prob = table.loc[table[elem] ==
                                     'T']['probability'].values[0]
                    if np.random.uniform(0, 1) < prob:
                        value = True
                    else:
                        value = False
                except Exception:
                    cols = table.columns[:-2]
                    values = tuple(df.iloc[i][cols])
                    values = ['T' if x is True else 'F' for x in values]

                    prob = table[(table[cols] == values).all(1)]
                    prob = prob['true_prob']
                    prob = prob.values[0]
                    if np.random.uniform(0, 1) < prob:
                        value = True
                    else:
                        value = False
                df.at[i, elem] = value
        return df

    def data_to_csv(self, df) -> None:
        df.to_csv("BayesianNetwork/data.csv", index=False)
