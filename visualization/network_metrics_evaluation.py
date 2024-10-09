import os
import pandas as pd
import networkx as nx
from abc import ABC, abstractmethod
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


class NetworkMetricStrategy(ABC):
    @abstractmethod
    def __init__(self, metric, year_list):
        self.year_list = year_list
        self.metric = metric

    def evaluate_network(self, att):
        results = dict()
        for year in self.year_list:
            data_path = os.path.abspath(os.path.join(PROJECT_ROOT, str(year), 'network_graph_phi.graphml'))
            G_phi = nx.read_graphml(data_path)
            metric_result = self.metric(G=G_phi, attribute=att)
            results[year] = metric_result
        self.results = results

    def get_results(self):
        return self.results


class NetworkMetricDirector():
    def __init__(self, year_list, att):
        self.year_list = year_list
        self.att = att
        self.builders = []
        self.builder.append(NetworkAssortativityStrategy(self.year_list))

    def construct(self):
        for builder in self.builders:
            builder.build_metric(self.year_list)


class NetworkMetricBuilder(ABC):
    def build_metric(self, year_list):
        pass


class NetworkAssortativityStrategy(NetworkMetricStrategy):
    def __init__(self, year_list):
        super().__init__(metric=nx.attribute_assortativity_coefficient, year_list=year_list)


if __name__ == '__main__':
    years = ['2016', '2018', '2019', '2020', '2021']
    random_i_d = True
    min_subjects = 5
    min_occurrences = 2
    net_types = ['high', 'less', 'all']
    for net_type in tqdm(net_types, desc='Calculating Metrics', unit='Net_Type'):
        folders = [f'{net_type}_{min_subjects}_{min_occurrences}_{year}' for year in years]
        if random_i_d:
            folders = [item if i == 0 else f"{item}_random{i}" for item in folders for i in range(len(folders) + 1)]

        df_results = pd.DataFrame(columns=['metric'] + folders)
        metric = NetworkAssortativityStrategy(folders)
        metric.evaluate_network(att="quantity_i-d")
        results = metric.get_results()
        results['metric'] = 'Assortativity_quantity_i-d'
        df_results.loc[len(df_results)] = results
        df_results.to_csv(f'Assortativity_{net_type}{"_random" if random_i_d else ""}.csv', index=False)

        print(results)
