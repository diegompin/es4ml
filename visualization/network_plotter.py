import os
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


class GraphPlotter:
    def __init__(self, graph, title, node_size, node_size_factor, edge_width, k, dpi):
        self.graph = graph
        self.title = title
        self.node_size = node_size
        self.node_size_factor = node_size_factor
        self.edge_width = edge_width
        self.k = k
        self.dpi = dpi

    def get_node_color(self, node):
        if self.graph.nodes[node]['quantity_i-d'] > 0:
            return 'red'

        return 'gray'
    
    def get_edge_color(self, edge):
        if self.graph.edges[edge]['transacoes_Id_entre_contas'] > 0:
            return 'red'

        return 'gray'

    def plot(self, save_path, random=None, pos=None):
        random = False if random is None else random
        # Get node colors and sizes
        node_colors = [self.get_node_color(n) for n in self.graph.nodes()]
        # node_sizes = [self.graph.degree(n) * self.node_size_factor if 'quantity_i-d' in self.graph.nodes[n] else self.node_size for n in self.graph.nodes()]
        node_sizes = [self.node_size_factor * self.node_size if self.graph.nodes[n][
                                                       'quantity_i-d'] > 0 else self.node_size
                      for n in self.graph.nodes()]
        node_sizes = [self.node_size * 1.11 if node < self.node_size else node for node in node_sizes]

        plt.figure(figsize=(12 * 4, 12 * 4))
        # Create layout and plot
        if pos is None:
            pos = nx.spring_layout(self.graph, k=self.k)

        nx.draw_networkx_nodes(self.graph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)
        nx.draw_networkx_edges(self.graph, pos, width=self.edge_width, alpha=0.3)
        plt.title(f'{self.title}', fontsize=40)
        plt.axis('off')

        # Save the plot
        plt.savefig(save_path, format=save_path.split('.')[-1], dpi=self.dpi, bbox_inches='tight')
        plt.clf()
        plt.close()

        if random:
            return pos


if __name__ == '__main__':
    # year = '2020'
    random_i_d = True
    how_many_randoms = 30
    min_subjects = 5
    min_occurrences = 2
    net_types = ['high', 'less', 'all']
    years = [str(year) for year in range(2019,2023)] + ['all']
    for net_type in tqdm(net_types, desc='Plotting Graphs', unit='Net_Type'):
        for year in tqdm(years, desc='Plotting Graphs', unit='Year'):
            path = os.path.join(
                PROJECT_ROOT, f'new-data_{net_type}_{min_subjects}_{min_occurrences}_{year}')

            if not os.path.exists(os.path.join(path, 'network_graph_phi.graphml')):
                continue

            G_phi = nx.read_graphml(os.path.join(path, 'network_graph_phi.graphml'))

            # print("Plotting Network.")
            plotter = GraphPlotter(G_phi, f'{year}', node_size=50, node_size_factor=2,
                                   edge_width=0.6, k=None, dpi=600)
            pos_nodes_edges = plotter.plot(f'{path}/phi_graph_output.pdf', random=random_i_d)

            if random_i_d:
                for i in range(1, how_many_randoms+1):
                    file_path = os.path.join(path, f'random{i}_network_graph_phi.graphml')
                    if not os.path.exists(file_path):
                        continue

                    G_phi = nx.read_graphml(file_path)

                    # print("Plotting Network.")
                    plotter = GraphPlotter(G_phi, f'{year}_random{i}', node_size=50,
                                           node_size_factor=2,
                                           edge_width=0.6, k=None, dpi=600)
                    plotter.plot(f'{path}/random{i}_phi_graph_output.pdf', pos=pos_nodes_edges)
            # print("Network plotted.")
