import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from os.path import exists
from os import makedirs
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


class DataLoader:
    def __init__(self, path):
        self.path = path

    def read_and_clean_data(self):
        df = pd.read_csv(self.path, delimiter=';', decimal=',', low_memory=False)

        # Clean data
        df['I-d'] = df['I-d'].apply(lambda x: '0' if pd.isna(x) else str(int(x)))
        df['CPF_CNPJ_OD'] = df['CPF_CNPJ_OD'].apply(lambda x: 'MISSING' if pd.isna(x) else str(x))
        df['NUMERO_CONTA'] = df['NUMERO_CONTA'].apply(lambda x: 'MISSING' if pd.isna(x) else str(x))
        return df


class GraphBuilder:
    def __init__(self):
        self.G = nx.Graph()

    def create_graph(self, df):
        mapping = {}
        node_id = 1
        self.G.clear()

        # Iterate through the dataframe rows
        for row in df:
            account_id = row[4]
            account_od_id = row[21]
            has_i_d = row[22]

            if account_id not in mapping:
                self.G.add_node(node_id, account=account_id)
                mapping[account_id] = node_id
                node_id += 1

            if account_od_id not in mapping and not np.isnan(account_od_id):
                self.G.add_node(node_id)
                mapping[account_od_id] = node_id
                node_id += 1

            # Add edges with 'i_d' based on condition
            if account_id in mapping and account_od_id in mapping:
                edge_condition = has_i_d == '1'
                self.G.add_edge(mapping[account_id], mapping[account_od_id], weight=1, i_d=edge_condition)

        return self.G


class GraphPlotter:
    def __init__(self, graph, cpf, node_size, node_size_factor, edge_width, k, dpi):
        self.graph = graph
        self.cpf = cpf
        self.node_size = node_size
        self.node_size_factor = node_size_factor
        self.edge_width = edge_width
        self.k = k
        self.dpi = dpi

    def get_node_color(self, node):
        if 'account' in self.graph.nodes[node]:
            return 'blue'
        # Check if the node without 'account' is connected to any edge where 'i_d' is True
        if any(edge_data['i_d'] for _, _, edge_data in self.graph.edges(node, data=True)):
            return '#EF5350'
        return 'gray'

    def plot(self, save_path):
        # Extract edge attributes
        edge_i_ds = [self.graph[u][v]['i_d'] for u, v in self.graph.edges()]
        edge_colors = ['#56C981' if i_d == False else '#EF5350' for i_d in edge_i_ds]

        # Get node colors and sizes
        node_colors = [self.get_node_color(n) for n in self.graph.nodes()]
        node_sizes = [self.graph.degree(n) * self.node_size_factor if 'account' in self.graph.nodes[n] else self.node_size for n in self.graph.nodes()]
        node_sizes = [self.node_size * 1.1 if node < self.node_size else node for node in node_sizes]

        # Create layout and plot
        pos = nx.spring_layout(self.graph, k=self.k)
        plt.figure(figsize=(12, 12))
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, width=self.edge_width, alpha=0.7)
        plt.title(f'CPF_CNPJ_TITULAR: {self.cpf}')
        plt.axis('off')

        # Save the plot
        plt.savefig(save_path, format=save_path.split('.')[-1], dpi=self.dpi, bbox_inches='tight')
        plt.clf()


def create_directory_if_not_exists(directory):
    if not exists(directory):
        makedirs(directory)


class GraphApplication:
    def __init__(self, data_loader, graph_builder, graph_plotter_class):
        self.data_loader = data_loader
        self.graph_builder = graph_builder
        self.graph_plotter_class = graph_plotter_class

    def run(self, cpf_cnpj_column, i_d_column, save_dir):
        create_directory_if_not_exists(save_dir)

        df = self.data_loader.read_and_clean_data()
        list_of_cpfs = list(set(df.loc[df[i_d_column] == '1', cpf_cnpj_column].to_list()))

        for cpf in tqdm(list_of_cpfs, desc='Processing graphs', unit='graph'):
            df_filtered = df[df[cpf_cnpj_column] == cpf].values
            graph = self.graph_builder.create_graph(df_filtered)

            plotter = self.graph_plotter_class(graph, cpf, node_size=30, node_size_factor=2, edge_width=1, k=0.12, dpi=600)
            plotter.plot(f'{save_dir}/graph_output_{cpf}.pdf')

            nx.write_graphml(graph, f'{save_dir}/graph_output_{cpf}.graphml')


path = os.path.abspath(os.path.join(PROJECT_ROOT, '../../..', 'pcpe', 'pcpe_02.csv'))
# path = '/data/pcpe_02.csv'
save_dir = 'visualization/outputs'
data_loader = DataLoader(path)
graph_builder = GraphBuilder()

graph_app = GraphApplication(data_loader, graph_builder, GraphPlotter)
graph_app.run(cpf_cnpj_column='CPF_CNPJ_TITULAR', i_d_column='I-d', save_dir=save_dir)
