import os
import zipfile
import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from itertools import compress
from abc import ABC, abstractmethod
from create_rr_phi_network import FileManager
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


class NetworkMetricBuilder():
    def build_metric(self, att):
        pass

class NetworkMetricStrategy(ABC, NetworkMetricBuilder):
    @abstractmethod    
    def __init__(self,metric,year_list):
        self.year_list = year_list
        self.metric = metric

    def build_metric(self, att):
        self.evaluate_network(att)

    def evaluate_network(self, att):
        results = dict()
        for year in self.year_list:
            data_path = os.path.abspath(os.path.join(PROJECT_ROOT,str(year), 'network_graph_phi.graphml'))
            G_phi = nx.read_graphml(data_path)
            if att: 
                metric_result = self.metric(G=G_phi,attribute=att)
            else:
                metric_result = self.metric(G=G_phi)
            results[year] = metric_result
        self.results = pd.DataFrame(results,index=[0])
        

    def get_results(self):
        return self.results

        

class NetworkMetricDirector():
    def __init__ (self, year_list, att):
        self.year_list = year_list
        self.att = att
        self.builders = []
        self.builders.append(NetworkAssortativityStrategy(self.year_list))
        self.builders.append(NetworkClusteringCoefficientStrategy(self.year_list))
    
    def construct(self):
        for builder in self.builders:
            builder.build_metric(self.att)



# class NetworkAssortativityStrategy(NetworkMetricStrategy):
#     def __init__(self, year_list):
#         super().__init__(metric=nx.attribute_assortativity_coefficient,year_list=year_list)
        

class NetworkAssortativityStrategy(NetworkMetricStrategy):
    def __init__(self, year_list):
        super().__init__(metric=nx.attribute_assortativity_coefficient,year_list=year_list)

class NetworkClusteringCoefficientStrategy(NetworkMetricStrategy):
    def __init__(self,year_list):
        super().__init__(metric=nx.clustering, year_list=year_list)

    def evaluate_network(self, att):
        return super().evaluate_network(att=None)
    
    

       
if __name__ == '__main__':
    year_list = [year for year in range(2015,2023)]
    att="quantity_i-d"
    director = NetworkMetricDirector(year_list=year_list, att=att)    
    director.construct()
    builders = director.builders
    for builder in builders:
        output_path = os.path.abspath(os.path.join(PROJECT_ROOT,'metric_results', type(builder).__name__+'.csv'))
        result = builder.get_results()
        result.to_csv(output_path,index=False)


 


