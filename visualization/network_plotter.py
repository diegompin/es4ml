import os
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from create_rr_phi_network import FileManager


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


class GraphPlotter:
    def __init__(self, node_size, node_size_factor, edge_width, k, dpi):
        self.graph = None
        self.title = None
        self.node_size = node_size
        self.node_size_factor = node_size_factor
        self.edge_width = edge_width
        self.k = k
        self.dpi = dpi
        self.attribute_node = None
        self.positions_of_nodes = None
        self.label_to_index = None
        self.color_map = {'I-d': 'red', 'I-e': '#03d7fc', 'IV-n': '#03fc2c'}

    def update_parameters(self, graph, title, attribute_node_):
        self.graph = graph
        self.title = title
        self.attribute_node = attribute_node_
        self.label_to_index = {data['conta']: node for node, data in graph.nodes(data=True) if 'conta' in data}

    def set_positions_of_nodes(self):
        positions_network = nx.spring_layout(self.graph, k=self.k)
        nodes = list(nx.get_node_attributes(self.graph, 'conta').values())
        self.positions_of_nodes = dict(zip(nodes, positions_network.values()))

    def get_positions_of_nodes(self):
        nodes = list(nx.get_node_attributes(self.graph, 'conta').values())
        list_index = {label: self.label_to_index.get(label) for label in nodes}
        return {list_index[node]: self.positions_of_nodes[node] for node in nodes}

    def get_node_color(self, node):
        if self.graph.nodes[node][self.attribute_node] > 0:
            # return self.color_map[self.attribute_node]
            return 'red'

        return 'gray'
    
    # def get_edge_color(self, edge):
    #     if self.graph.edges[edge][f'transacoes_{self.attribute.replace("-", "_").lower()}_entre_contas'] > 0:
    #         return self.color_map[self.attribute]
    #
    #     return 'gray'

    def plot(self, save_path):
        # Get node colors and sizes
        node_colors = [self.get_node_color(n) for n in self.graph.nodes()]
        # node_sizes = [self.graph.degree(n) * self.node_size_factor if 'quantity_iv-n' in self.graph.nodes[n] else self.node_size for n in self.graph.nodes()]
        # node_sizes = [self.node_size_factor * self.node_size if self.graph.nodes[n][
        #                                                             f'quantity_{self.typology.lower()}'] > 0 else self.node_size
        #               for n in self.graph.nodes()]
        node_sizes = [self.node_size_factor * self.node_size * (self.graph.nodes[n][f'betweenness_node']*100) if self.graph.nodes[n][
                                                                    f'betweenness_node'] > 0 else self.node_size
                      for n in self.graph.nodes()]
        node_sizes = [self.node_size * 1.01 if node <= self.node_size else node for node in node_sizes]

        plt.figure(figsize=(12 * 4, 12 * 4))
        # Create layout and plot

        labels = nx.get_node_attributes(self.graph, 'conta')
        nx.draw_networkx_nodes(
            self.graph, self.get_positions_of_nodes(), node_size=node_sizes, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(self.graph, self.get_positions_of_nodes(), width=self.edge_width, alpha=0.3)
        nx.draw_networkx_labels(self.graph, self.get_positions_of_nodes(), labels=labels, font_size=6)
        plt.title(f'{self.title}', fontsize=40)
        plt.axis('off')

        # Save the plot
        plt.savefig(save_path, format=save_path.split('.')[-1], dpi=self.dpi, bbox_inches='tight')
        # plt.savefig(save_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
        # plt.savefig(save_path.replace('.pdf', '.svg'), format='svg', dpi=self.dpi, bbox_inches='tight')
        plt.clf()
        plt.close()


if __name__ == '__main__':
    random_i_d = True
    attribute = 'involvement'
    network_filename = 'less_EGO_network_graph_phi'
    how_many_randoms = 30
    min_subjects = 5
    min_occurrences = 2
    net_types = ['high']
    years = ['all'] + [str(year) for year in range(2019,2023)]
    plotter = GraphPlotter(node_size=50, node_size_factor=15, edge_width=0.6, k=None, dpi=300)
    folders = [os.path.join('outputs', f'{year}_{net_type}_{min_subjects}_{min_occurrences}') for year in years for
             net_type in net_types]

    for folder in tqdm(folders, desc='Creating random networks'):
        plot_output_folder = os.path.join(folder, 'plots')
        FileManager.create_dir(plot_output_folder)

        if not os.path.exists(os.path.join(folder, f'{network_filename}.gml')):
            continue

        G_phi = nx.read_gml(os.path.join(folder, f'{network_filename}.gml'))

        plotter.update_parameters(G_phi, f'{os.path.basename(os.path.normpath(folder))}', attribute)
        if 'all' in folder:
            plotter.set_positions_of_nodes()

        plotter.plot(os.path.join(plot_output_folder, f'{network_filename}.pdf'))

        if random_i_d:
            for i in range(1, how_many_randoms+1):
                file_path = os.path.join(folder, 'random_networks', f'random{i}_{network_filename}.gml')
                if not os.path.exists(file_path):
                    continue

                G_phi = nx.read_gml(file_path)

                plotter.update_parameters(G_phi, f'random{i}_{os.path.basename(os.path.normpath(folder))}', attribute)

                plotter.plot(os.path.join(plot_output_folder, f'random{i}_{network_filename}.pdf'))
