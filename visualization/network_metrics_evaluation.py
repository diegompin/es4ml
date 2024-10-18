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

    def evaluate_network(self, att, random, random_num):
        results = dict()
        for year in self.year_list:
            data_path = os.path.abspath(os.path.join(PROJECT_ROOT, str(year), 'network_graph_phi.graphml'))
            G_phi = nx.read_graphml(data_path)
            metric_result = self.metric(G=G_phi, attribute=att)
            results[year] = metric_result
            if random:
                for i in range(1, random_num+1):
                    data_path = os.path.abspath(os.path.join(PROJECT_ROOT, str(year), f'random{i}_network_graph_phi.graphml'))
                    G_phi = nx.read_graphml(data_path)
                    metric_result = self.metric(G=G_phi, attribute=att)
                    results[f'{year}_random{i}'] = metric_result
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
    random_i_d = True
    typologies = ['I-d', 'I-e', 'IV-n']
    how_many_randoms = 30
    min_subjects = 5
    min_occurrences = 2
    net_types = ['high']
    years = ['all'] + [str(year) for year in range(2019,2023)]
    for typology in tqdm(typologies, desc='Calculating Metrics', unit='Typology'):
        for net_type in tqdm(net_types, desc='Calculating Metrics', unit='Net_Type'):
            folders = [f'new-data_{typology}_{net_type}_{min_subjects}_{min_occurrences}_{year}' for year in years]

            metric = NetworkAssortativityStrategy(folders)
            metric.evaluate_network(att=f"quantity_{typology.lower()}", random=random_i_d, random_num=how_many_randoms)
            results = metric.get_results()
            df_results = pd.DataFrame(list(results.items()), columns=['network', f'Assortativity_quantity_{typology.lower()}'])

            metric.evaluate_network(att="prevalence", random=random_i_d, random_num=how_many_randoms)
            results = metric.get_results()
            df_results['Assortativity_prevalence'] = df_results['network'].map(results)

            metric.evaluate_network(att="sum_metric", random=random_i_d, random_num=how_many_randoms)
            results = metric.get_results()
            df_results['Assortativity_sum-metric'] = df_results['network'].map(results)

            df_results.to_csv(f'Assortativity_{typology}_{net_type}.csv', index=False)

            # print(results)
