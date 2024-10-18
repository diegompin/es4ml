from cdlib import NodeClustering
import pandas as pd
import networkx as nx
import cdlib
from cdlib.viz import plot_network_clusters 
import os.path
from cdlib.algorithms import girvan_newman
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


class CommunityManager:

    def __init__(self, cluster:NodeClustering, title):
        self.cluster = cluster
        self.title = title


    def aggregate_nodes(self):
        self.aggregation = dict()
        communities = self.cluster.communities
        for i,community in enumerate(communities):
            community_id = i+1
            if len(community) != 1:  
                for node in community:
                    self.aggregation[node] = f'community{community_id}'
            else:
                self.aggregation[community[0]] = 'singleton'


        self.df_node_aggregation = pd.DataFrame.from_dict(self.aggregation,orient='index')
        output_path = os.path.join(PROJECT_ROOT,f'communities_id/{self.title}.csv')
        self.df_node_aggregation.to_csv(output_path)




if __name__ == '__main__':
    network_path = os.path.join(PROJECT_ROOT,'I-d_results_data-2.0/new-data_high_5_2_all/network_graph_phi.graphml')
    G_network_all = nx.read_graphml(network_path)
    partition_all = girvan_newman(g_original=G_network_all, level=1)
    com_man = CommunityManager(cluster=partition_all, title='network_all')
    com_man.aggregate_nodes()

