from cdlib import NodeClustering
import pandas as pd
import networkx as nx
import cdlib
from cdlib.viz import plot_network_clusters 
import os.path
from cdlib.algorithms import girvan_newman
from networkx import Graph
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


class CommunityManager:

    def __init__(self, cluster:NodeClustering,G:Graph, title):
        self.cluster = cluster
        self.G = G
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

    
    def set_network_community_id(self):
        nx.set_node_attributes(G=self.G,values=self.aggregation,name='community_id')
        output_path = os.path.join(PROJECT_ROOT,'communities_id','graph_exports',f'{self.title}_graph.graphml')
        nx.write_graphml(G=self.G, path=output_path)




if __name__ == '__main__':
    typologies = ['I-d_results_data-2.0','I-e_results_data-2.0','IV-n_results_data-2.0']
    networks = ['new-data_high_5_2_2019','new-data_high_5_2_2020','new-data_high_5_2_2021','new-data_high_5_2_2022','new-data_high_5_2_all']
    target_graph = 'network_graph_phi.graphml'
    for typology in typologies:
        for network in networks:
            network_path = os.path.join(PROJECT_ROOT,typology,network, target_graph)
            G_network = nx.read_graphml(network_path)
            partition = girvan_newman(g_original=G_network, level=1)
            title = f"{typology.strip('data-2.0')}{network.replace('new-data_','').replace('_5_2','')}"
            com_man = CommunityManager(G=G_network, cluster=partition,title= title)
            com_man.aggregate_nodes()
            com_man.set_network_community_id()         
    
    
    
    #network_path = os.path.join(PROJECT_ROOT,'I-d_results_data-2.0/new-data_high_5_2_all/network_graph_phi.graphml')
    #G_network_all = nx.read_graphml(network_path)
    #partition_all = girvan_newman(g_original=G_network_all, level=1)
    #com_man = CommunityManager(cluster=partition_all, title='network_all')
    #com_man.aggregate_nodes()

