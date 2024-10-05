import os
import zipfile
import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from itertools import compress
from abc import ABC, abstractmethod
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

class NetworkMetricStrategy(ABC):
    @abstractmethod    
    def __init__(self,metric,year_list):
        self.year_list = year_list
        self.metric = metric

    def evaluate_network(self, att):
        results = dict()
        for year in self.year_list:
            data_path = os.path.abspath(os.path.join(PROJECT_ROOT,str(year), 'network_graph_phi.graphml'))
            G_phi = nx.read_graphml(data_path)
            metric_result = self.metric(G=G_phi,attribute=att)
            results[year] = metric_result
        self.results = results

    def get_results(self):
        return self.results

        

class NetworkMetricDirector():
    def __init__ (self, year_list, att):
        self.year_list = year_list
        self.att = att
        self.builders = []
        self.builder.append(NetworkAssortativityStrategy(self.year_list))
    
    def construct(self):
        for builder in self.builders:
            builder.build_metric(self.year_list)

class NetworkMetricBuilder(ABC):
    def build_metric(self,year_list):
        pass

class NetworkAssortativityStrategy(NetworkMetricStrategy):
    def __init__(self, year_list):
        super().__init__(metric=nx.attribute_assortativity_coefficient,year_list=year_list)
        

    

       
if __name__ == '__main__':
    year_list = [year for year in range(2015,2023)]
    metric = NetworkAssortativityStrategy(year_list)
    metric.evaluate_network(att="quantity_i-d")
    results = metric.get_results()
    print(results)


 


