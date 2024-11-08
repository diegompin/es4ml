import os
import warnings
import networkx as nx

from tqdm import tqdm
from create_rr_phi_network import DataPreparer, FileManager

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.abspath(os.path.join(PROJECT_ROOT, '../../..', 'pcpe'))

net_type = 'high'
min_subjects = 5
min_occurrences = 1
# year = 'all'

# output_path = os.path.join(PROJECT_ROOT, f'new-data_{typology}_{net_type}_{min_subjects}_{min_occurrences}_{year}')
#
# G = nx.read_gml(os.path.join(output_path, 'network_graph_phi.gml'))


years = ['all'] + [str(year) for year in range(2019, 2023)]

for year in tqdm(years, desc='Creating Graphs Years', unit='Year', position=0):

    output_path = os.path.join(PROJECT_ROOT, 'outputs', f'{year}_{net_type}_{min_subjects}_{min_occurrences}')

    G = nx.read_gml(os.path.join(output_path, 'network_graph_phi.gml'))

    # df = DataPreparer.read_data(data_path, year)
    #
    # target_nodes = df['CONTA_TITULAR'].unique().tolist()
    #
    # matched_nodes = [node for node, data in G.nodes(data=True) if data.get("conta") in target_nodes]

    matched_nodes = [node for node, data in G.nodes(data=True) if data.get("is_conta_titular")]

    for node in matched_nodes:
        neighbors = list(G.neighbors(node))
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                G.add_edge(neighbors[i], neighbors[j], weight=0.01)


    # for u, v, data in G.edges(data=True):
    #     if data.get('weight', 0) > 0:
    #         data['weight'] = 1


    node_bet = nx.betweenness_centrality(G)
    nx.set_node_attributes(G, node_bet, 'betweenness_node')

    edge_bet = nx.edge_betweenness_centrality(G)
    nx.set_edge_attributes(G, edge_bet, 'betweenness_edge')

    FileManager.write_gml(G, os.path.join(output_path, f'less_EGO_network_graph_phi.gml'))

    # G.remove_nodes_from(matched_nodes)
    #
    # node_betweenness = nx.betweenness_centrality(G, weight=None)
    # nx.set_node_attributes(G, node_betweenness, 'betweenness_node')
    #
    # edge_betweenness = nx.edge_betweenness_centrality(G, weight=None)
    # nx.set_edge_attributes(G, edge_betweenness, 'betweenness_edge')
    #
    # FileManager.write_gml(G, os.path.join(output_path, f'no_EGO_2022.gml'))
