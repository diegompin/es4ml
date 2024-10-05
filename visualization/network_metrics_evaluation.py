import os
import zipfile
import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from itertools import compress
from abc import ABC, abstractmethod
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


class NetworkMetricEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_network(self, att):
        pass
        

class NetworkAssortativityStrategy(NetworkMetricEvaluationStrategy):

    def __init__(self, year_list):
        self.year_list = year_list

    def evaluate_network(self, att):
        coefficients = dict()
        for year in self.year_list:
            G_phi = nx.read_graphml(str(year) + '/network_graph_phi.graphml')
            assortativity = nx.attribute_assortativity_coefficient(G=G_phi,attribute=att)
            coefficients[year] = assortativity
        return coefficients    


if __name__ == '__main__':
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.abspath(os.path.join(PROJECT_ROOT))
    year_list = [year for year in range(2015,2021)]
    metric = NetworkAssortativityStrategy(year_list)
    coefficients = metric.evaluate_network(att="quantity_i-d")


 


