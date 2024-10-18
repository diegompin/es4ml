import os
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


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
        self.typology = None
        self.positions_of_nodes = None
        self.label_to_index = None
        self.color_map = {'I-d': 'red', 'I-e': '#03d7fc', 'IV-n': '#03fc2c'}

    def update_parameters(self, graph, title, typology_):
        self.graph = graph
        self.title = title
        self.typology = typology_
        self.label_to_index = {data['label']: node for node, data in graph.nodes(data=True) if 'label' in data}

    def set_positions_of_nodes(self):
        positions_network = nx.spring_layout(self.graph, k=self.k)
        nodes = list(nx.get_node_attributes(self.graph, 'label').values())
        self.positions_of_nodes = dict(zip(nodes, positions_network.values()))

    def get_positions_of_nodes(self):
        nodes = list(nx.get_node_attributes(self.graph, 'label').values())
        list_index = {label: self.label_to_index.get(label) for label in nodes}
        return {list_index[node]: self.positions_of_nodes[node] for node in nodes}

    def get_node_color(self, node):
        if self.graph.nodes[node][f'quantity_{self.typology.lower()}'] > 0:
            return self.color_map[self.typology]

        return 'gray'
    
    def get_edge_color(self, edge):
        if self.graph.edges[edge][f'transacoes_{self.typology.replace("-", "")}_entre_contas'] > 0:
            return self.color_map[self.typology]

        return 'gray'

    def plot(self, save_path):
        # Get node colors and sizes
        node_colors = [self.get_node_color(n) for n in self.graph.nodes()]
        # node_sizes = [self.graph.degree(n) * self.node_size_factor if 'quantity_iv-n' in self.graph.nodes[n] else self.node_size for n in self.graph.nodes()]
        node_sizes = [self.node_size_factor * self.node_size if self.graph.nodes[n][
                                                                    f'quantity_{self.typology.lower()}'] > 0 else self.node_size
                      for n in self.graph.nodes()]
        node_sizes = [self.node_size * 1.11 if node < self.node_size else node for node in node_sizes]

        plt.figure(figsize=(12 * 4, 12 * 4))
        # Create layout and plot

        labels = nx.get_node_attributes(self.graph, 'label')
        nx.draw_networkx_nodes(
            self.graph, self.get_positions_of_nodes(), node_size=node_sizes, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(self.graph, self.get_positions_of_nodes(), width=self.edge_width, alpha=0.3)
        nx.draw_networkx_labels(self.graph, self.get_positions_of_nodes(), labels=labels, font_size=6)
        plt.title(f'{self.title}', fontsize=40)
        plt.axis('off')

        # Save the plot
        plt.savefig(save_path, format=save_path.split('.')[-1], dpi=self.dpi, bbox_inches='tight')
        # plt.savefig(save_path.replace('.pdf', '.png'), format='png', dpi=self.dpi, bbox_inches='tight')
        plt.savefig(save_path.replace('.pdf', '.svg'), format='svg', dpi=self.dpi, bbox_inches='tight')
        plt.clf()
        plt.close()


if __name__ == '__main__':
    # year = '2020'
    random_i_d = True
    typologies = ['I-d', 'I-e', 'IV-n']
    how_many_randoms = 30
    min_subjects = 5
    min_occurrences = 2
    net_types = ['high']
    years = ['all'] + [str(year) for year in range(2019,2023)]
    plotter = GraphPlotter(node_size=50, node_size_factor=2, edge_width=0.6, k=None, dpi=600)
    for typology in tqdm(typologies, desc='Plotting Graphs Typologies', unit='Typology', position=0):
        for net_type in tqdm(net_types, desc='Plotting Graphs Net_Types', unit='Net_Type', position=1, leave=False):
            for year in tqdm(years, desc='Plotting Graphs Years', unit='Year', position=2, leave=False):
                path = os.path.join(
                    PROJECT_ROOT, f'new-data_{typology}_{net_type}_{min_subjects}_{min_occurrences}_{year}')

                if not os.path.exists(os.path.join(path, 'network_graph_phi.graphml')):
                    continue

                G_phi = nx.read_graphml(os.path.join(path, 'network_graph_phi.graphml'))

                # print("Plotting Network.")
                plotter.update_parameters(G_phi, f'{year}', typology)
                if year == "all" and typology == 'I-d':
                    plotter.set_positions_of_nodes()

                plotter.plot(f'{path}/phi_graph_output.pdf')

                if random_i_d:
                    for i in range(1, how_many_randoms+1):
                        file_path = os.path.join(path, f'random{i}_network_graph_phi.graphml')
                        if not os.path.exists(file_path):
                            continue

                        G_phi = nx.read_graphml(file_path)

                        # print("Plotting Network.")
                        plotter.update_parameters(G_phi, f'{year}_random{i}', typology)

                        plotter.plot(f'{path}/random{i}_phi_graph_output.pdf')
                # print("Network plotted.")
