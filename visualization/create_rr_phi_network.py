import os
import zipfile
import warnings
import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from itertools import compress
from abc import ABC, abstractmethod
from tqdm import tqdm


warnings.filterwarnings("ignore", category=RuntimeWarning)


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


class FileManager:
    """Single Responsibility for managing file operations."""
    @staticmethod
    def create_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    @staticmethod
    def write_gml(graph, path):
        nx.write_gml(graph, path)

    @staticmethod
    def save_arrays_to_zip_as_csv(arrays, filenames, dir_path, zip_filename='arrays.zip'):
        zip_path = os.path.join(dir_path, zip_filename)

        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            for array, filename in zip(arrays, filenames):
                csv_path = str(os.path.join(dir_path, filename + '.csv'))
                np.savetxt(csv_path, np.asarray(array), delimiter=',', fmt='%.2f')
                zipf.write(csv_path, arcname=filename + '.csv')
                os.remove(csv_path)


class DataPreparer:
    @staticmethod
    def read_data(data_path, year=None):
        year = '' if year is None or year in ['', 'all'] else year

        dtype_dict = {
            'NUMERO_CASO': 'str',
            'NUMERO_BANCO': 'str',
            'NOME_BANCO': 'str',
            'NUMERO_AGENCIA': 'str',
            'NUMERO_CONTA': 'str',
            'TIPO': 'str',
            'CPF_CNPJ_TITULAR': 'str',
            'NOME_TITULAR': 'str',
            'DATA_LANCAMENTO': 'str',
            'CPF_CNPJ_OD': 'str',
            'NOME_PESSOA_OD': 'str',
            'CNAB': 'str',
            'DESCRICAO_LANCAMENTO': 'str',
            'VALOR_TRANSACAO': 'float64',
            'NATUREZA_LANCAMENTO': 'str',
            'I-d': 'str',
            'I-e': 'str',
            'IV-n': 'str',
            'RAMO_ATIVIDADE_1': 'str',
            'RAMO_ATIVIDADE_2': 'str',
            'RAMO_ATIVIDADE_3': 'str',
            'LOCAL_TRANSACAO': 'str',
            'NUMERO_DOCUMENTO': 'str',
            'NUMERO_DOCUMENTO_TRANSACAO': 'str',
            'VALOR_SALDO': 'float64',
            'NATUREZA_SALDO': 'str',
            'NUMERO_BANCO_OD': 'str',
            'NUMERO_AGENCIA_OD': 'str',
            'NUMERO_CONTA_OD': 'str',
            'NOME_ENDOSSANTE_CHEQUE': 'str',
            'DOC_ENDOSSANTE_CHEQUE': 'str',
            'DIA_LANCAMENTO': 'str',
            'MES_LANCAMENTO': 'str',
            'ANO_LANCAMENTO': 'str'
        }

        df = pd.read_csv(os.path.join(data_path, 'pcpe_03.csv'), sep=';', decimal=',', dtype=dtype_dict)
        if year:
            df = df.loc[df['ANO_LANCAMENTO'] == year]

        # df['CONTA_TITULAR'] = df['NUMERO_AGENCIA'] + '_' + df['NUMERO_CONTA']
        # df['CONTA_OD'] = df['NUMERO_AGENCIA_OD'] + '_' + df['NUMERO_CONTA_OD']
        df['CONTA_TITULAR'] = df['NUMERO_BANCO'] + '_' + df['NUMERO_AGENCIA'] + '_' + df['NUMERO_CONTA']
        df['CONTA_OD'] = df['NUMERO_BANCO_OD'] + '_' + df['NUMERO_AGENCIA_OD'] + '_' + df['NUMERO_CONTA_OD']
        df['CONTA_OD'] = df['CONTA_OD'].fillna('EMPTY')
        df.loc[df['CONTA_OD'] == '0_0', 'CONTA_OD'] = 'EMPTY'

        df = df.reset_index(drop=True)

        return df

    """Handles data preparation for network analysis."""
    @staticmethod
    def prepare_dataframe_for_cooccurrence(df, year=None, typology=None):
        year = '' if year is None or year in ['', 'all'] else year
        typology = ['I-d'] if typology is None else typology

        df[typology] = df[typology].fillna('0')
        if year:
            df = df.loc[df['ANO_LANCAMENTO'] == year]

        contas_origem = pd.Index(df['CONTA_TITULAR'].unique())
        contas_destino = pd.Index(df['CONTA_OD'].unique())
        contas = np.concatenate((contas_origem, contas_destino[~np.isin(contas_destino, contas_origem)])).tolist()

        return contas, DataPreparer.create_sparse_matrix(df, contas)

    @staticmethod
    def create_dataframes_for_attr(df, year=None, typology=None):
        year = '' if year is None or year in ['', 'all'] else year
        typology = ['I-d'] if typology is None else typology

        df[typology] = df[typology].fillna('0')
        if year:
            df = df.loc[df['ANO_LANCAMENTO'] == year]

        df_origem = df[['CONTA_TITULAR'] + typology].rename(columns={'CONTA_TITULAR': 'CONTA'})
        df_destino = df[['CONTA_OD'] + typology].rename(columns={'CONTA_OD': 'CONTA'})
        df_conta_id = pd.concat([df_origem, df_destino])
        df_conta_id = df_conta_id.astype({key: 'int32' for key in typology})
        df_count = df_conta_id.groupby(['CONTA']).sum().reset_index()
        df_pair_count = df[['CONTA_TITULAR', 'CONTA_OD'] + typology].astype({key: 'int32' for key in typology}).groupby(['CONTA_TITULAR', 'CONTA_OD']).sum().reset_index()

        return df_count, df_pair_count

    @staticmethod
    def create_attr_for_nodes_edges(metric, P, L, df_count, df_pair_count, typologies_, data_final_typology_, data):
        # Filter `df_count` once at the start
        df_count = df_count[df_count['CONTA'].isin(L)].copy()
        data_final_typology_ = data_final_typology_[data_final_typology_['CONTA_TITULAR'].isin(L)].copy()
        df_pair_count = df_pair_count[
            df_pair_count['CONTA_TITULAR'].isin(L) & df_pair_count['CONTA_OD'].isin(L)
            ].copy()
        list_conta_titular = data.loc[data['CONTA_TITULAR'].isin(L), 'CONTA_TITULAR'].unique().tolist()

        # Calculate metric sums
        metric_sums = np.sum(metric, axis=1)

        # Map `L` list to index for faster lookups
        L_index_map = {label: i for i, label in enumerate(L)}

        # Prepare node attributes
        node_attrs_1 = {
            i: {
                'prevalence': int(P[i]),
                'sum_metric': float(metric_sums[i]),
                'conta': L[i],
                'is_conta_titular': True if L[i] in list_conta_titular else False
            }
            for i in range(len(L))
        }

        df_count_dict = df_count.set_index('CONTA')[typologies_].to_dict(orient='index')

        node_attrs_2 = {
            i: {
                f'quantity_{typology.replace("-", "_").lower()}': int(df_count_dict.get(L[i], {}).get(typology, 0))
                for typology in typologies_
            }
            for i in range(len(L))
        }

        node_attrs_3 = {
            i: {
                'quantity_of_bad_transactions': sum(df_count_dict.get(L[i], {}).values())
            }
            for i in range(len(L))
        }

        data_final_typology_dict = data_final_typology_.set_index('CONTA_TITULAR')[typologies_].to_dict(orient='index')
        involvement_class = {
            tuple({}.items()): '0 - NO_INVOLVEMENT',
            tuple({'I-d': '0', 'I-e': '0', 'IV-n': '0'}.items()): '0 - NO_INVOLVEMENT',
            tuple({'I-d': '1', 'I-e': '0', 'IV-n': '0'}.items()): '1 - I-D INVOLVEMENT',
            tuple({'I-d': '0', 'I-e': '1', 'IV-n': '0'}.items()): '2 - I-E INVOLVEMENT',
            tuple({'I-d': '0', 'I-e': '0', 'IV-n': '1'}.items()): '3 - IV-N INVOLVEMENT',
            tuple({'I-d': '1', 'I-e': '1', 'IV-n': '0'}.items()): '4 - I-D AND I-E INVOLVEMENT',
            tuple({'I-d': '1', 'I-e': '0', 'IV-n': '1'}.items()): '5 - I-D AND IV-N INVOLVEMENT',
            tuple({'I-d': '0', 'I-e': '1', 'IV-n': '1'}.items()): '6 - I-E AND IV-N INVOLVEMENT',
            tuple({'I-d': '1', 'I-e': '1', 'IV-n': '1'}.items()): '7 - I-D AND I-E AND IV-N INVOLVEMENT'
        }

        node_attrs_4 = {
            i: {
                'involvement': int(any(j == '1' for j in list(data_final_typology_dict.get(L[i], {}).values()))),
                'involvement_class': involvement_class[tuple(data_final_typology_dict.get(L[i], {}).items())]
            }
            for i in range(len(L))
        }

        node_attrs = {i: {**node_attrs_1.get(i, {}), **node_attrs_2.get(i, {}), **node_attrs_3.get(i, {}), **node_attrs_4.get(i, {})} for i in range(len(L))}

        df_pair_count['idx_origem'] = df_pair_count['CONTA_TITULAR'].map(L_index_map)
        df_pair_count['idx_destino'] = df_pair_count['CONTA_OD'].map(L_index_map)

        edge_attrs = {
            (row['idx_origem'], row['idx_destino']): {
                f'transacoes_{typology.replace("-", "_").lower()}_entre_contas': row[typology]
                for typology in typologies_
            }
            for _, row in df_pair_count.iterrows()
        }

        return node_attrs, edge_attrs

    @staticmethod
    def read_final_typology_label(data_path_):
        dtype_dict = {
            'NUM_BANCO': 'str',
            'NOME_BANCO': 'str',
            'NUM_AGENCIA': 'str',
            'NUM_CONTA': 'str',
            'I-d': 'str',
            'I-e': 'str',
            'IV-n': 'str',
        }

        df = pd.read_csv(os.path.join(data_path_, 'POSITIVE_ACCOUNTS.csv'), sep=';', dtype=dtype_dict)
        df['CONTA_TITULAR'] = df['NUM_BANCO'] + '_' + df['NUM_AGENCIA'] + '_' + df['NUM_CONTA']

        return df[['CONTA_TITULAR', 'I-d', 'I-e', 'IV-n']]


    @staticmethod
    def create_sparse_matrix(df, contas):#(132,379)
        contas_dict = {conta: idx for idx, conta in enumerate(contas)}
        origem_factors = df['CONTA_TITULAR'].map(contas_dict).values
        destino_factors = df['CONTA_OD'].map(contas_dict).values

        n_rows = df.shape[0]
        n_cols = len(contas)

        row_indices = np.concatenate([np.arange(n_rows), np.arange(n_rows)])
        col_indices = np.concatenate([origem_factors, destino_factors])
        data = np.ones(len(row_indices))
        data[len(origem_factors):][destino_factors == contas_dict['EMPTY']] = 0

        return coo_matrix((data, (row_indices, col_indices)), shape=(n_rows, n_cols)).tocsr()


class IGraphStrategy(ABC):
    """Interface Segregation: Abstract class for graph generation strategies."""
    @abstractmethod
    def create_graph(self, matrix, node_attrs, edge_attrs):
        pass


class DefaultGraphStrategy(IGraphStrategy):
    """Concrete implementation of graph generation."""
    def create_graph(self, matrix, node_attrs, edge_attrs):
        G = nx.from_numpy_array(matrix)
        self.put_attributes(G, node_attrs, edge_attrs)
        return G

    @staticmethod
    def put_attributes(G, node_attrs, edge_attrs):
        nx.set_node_attributes(G, node_attrs)
        nx.set_edge_attributes(G, edge_attrs)

        node_betweenness = nx.betweenness_centrality(G, weight=None)
        node_clustering = nx.clustering(G,weight = None)
        node_degree = dict(G.degree(weight=None))

        nx.set_node_attributes(G, node_betweenness, 'betweenness_node')

        nx.set_node_attributes(G,node_clustering,'clustering_node')
        nx.set_node_attributes(G,node_degree,'degree_node')

        edge_betweenness = nx.edge_betweenness_centrality(G, weight=None)
        nx.set_edge_attributes(G, edge_betweenness, 'betweenness_edge')


# class DefaultGraphRandomStrategy(IGraphStrategy):
#     """Concrete implementation of graph generation."""
#     def create_graph(self, matrix, P, L, df_count, df_pair_count, typology):
#         G = nx.from_numpy_array(matrix)
#         self.put_attributes(G, P, L, matrix, df_count, df_pair_count, typology)
#         return G
#
#     def put_attributes(self, G, P, L, metric, df_count, df_pair_count, typology):
#         df_count = df_count[df_count['CONTA'].isin(L)]
#         df_count = self.randomize_attribute(df_count, '1')
#         # filtered_df_pair_count = df_pair_count[
#         #     df_pair_count['CONTA_TITULAR'].isin(L) | df_pair_count['CONTA_OD'].isin(L)
#         # ]
#
#         metric_sums = np.sum(metric, axis=1)
#
#         node_attrs = {
#             i: {
#                 'prevalence': int(P[i]),
#                 'sum_metric': float(metric_sums[i]),
#                 'label': L[i],
#                 f'quantity_{typology.lower()}': int(df_count.loc[df_count['CONTA'] == L[i], '1'].values[0])
#                 if not df_count[df_count['CONTA'] == L[i]].empty else 0
#             }
#             for i in G.nodes
#         }
#         nx.set_node_attributes(G, node_attrs)
#
#         edge_attrs = {}
#         for _, row in df_pair_count.iterrows():
#             conta_origem = row['CONTA_TITULAR']
#             conta_destino = row['CONTA_OD']
#             valor_coluna_1 = row['1']
#
#             if conta_origem in L and conta_destino in L:
#                 idx_origem = L.index(conta_origem)
#                 idx_destino = L.index(conta_destino)
#
#                 edge_attrs[(idx_origem, idx_destino)] = {f'transacoes_{typology.replace("-", "")}_entre_contas': valor_coluna_1}
#         nx.set_edge_attributes(G, edge_attrs)
#
#         node_betweenness = nx.betweenness_centrality(G, weight=None)
#         nx.set_node_attributes(G, node_betweenness, 'betweenness_node')
#
#         edge_betweenness = nx.edge_betweenness_centrality(G, weight=None)
#         nx.set_edge_attributes(G, edge_betweenness, 'betweenness_edge')
#
#     @staticmethod
#     def randomize_attribute(df, column_name):
#         column_values = df[column_name].values
#
#         num_zeros = np.sum(column_values == 0)
#         num_non_zeros = len(column_values) - num_zeros
#
#         new_non_zero_values = np.random.randint(1, np.max(column_values) + 1, size=num_non_zeros)
#         randomized_values = np.concatenate([np.zeros(num_zeros, dtype=column_values.dtype), new_non_zero_values])
#         np.random.shuffle(randomized_values)
#
#         df.loc[:, column_name] = randomized_values
#         return df


class NetworkCoOccurrence:

    def get_network(self, labels, occurrences, min_subjects_=5, min_occurrences_=2, net_type_=None):
        C, CC, L = self.get_cooccurrence(occurrences.copy(), labels, min_subjects_, min_occurrences_)
        N = C.shape[0]

        RR_dist, RR_graph = self.calculate_risk_ratio(CC, N, net_type_)

        P = np.diag(CC.toarray())

        Phi_dist, Phi_graph = self.calculate_phi(CC, N, net_type_)

        return C, CC, RR_graph, RR_dist, Phi_graph, Phi_dist, P, L

    def get_cooccurrence(self, occurrence, L, min_subjects, min_occurrences):
        column_sums = occurrence.sum(axis=0).A1
        col_mask = column_sums > min_subjects
        C = occurrence[:, col_mask]
        L = list(compress(L, col_mask))

        row_sums = C.sum(axis=1).A1
        row_mask = row_sums >= min_occurrences
        C = C[row_mask, :]

        CC = C.T @ C
        return C, CC, L

    def product_matrix(self, V):
        return np.float64(V[:, np.newaxis] * V)

    def get_coprevalence(self, P):
        P_cooccurrence = np.maximum(P[:, np.newaxis], P[np.newaxis, :])
        return P_cooccurrence

    def calculate_risk_ratio(self, CC, N, net_type):
        RR, RR_l, RR_u = self.get_risk_ratio(CC, N)
        RR_graph, RR_dist = self.get_graph_sig(RR, RR_l, RR_u)
        if net_type != 'all':
            if net_type == 'high':
                RR_graph[RR_graph <= 1] = 0
                RR_dist = RR_dist[RR_dist > 1]
            if net_type == 'less':
                RR_graph[RR_graph >= 1] = 0
                RR_graph = 1 / RR_graph
                RR_graph[~np.isfinite(RR_graph)] = 0
                RR_dist = RR_dist[RR_dist < 1]
        return RR_dist, RR_graph

    def get_risk_ratio(self, CC, N):
        P = np.diagonal(CC.toarray())
        PP = self.product_matrix(P)

        RR = N * CC.toarray() / PP
        RR[~np.isfinite(RR)] = 0

        SIG = (1 / CC.toarray()) + (1 / PP)
        if N == 0:
            SIG = SIG * np.inf
        else:
            SIG = SIG - 1 / N - 1 / (N ** 2)

        SIG[~np.isfinite(SIG)] = 0
        RR_l = RR * np.exp(-2.56 * SIG)
        RR_u = RR * np.exp(+2.56 * SIG)
        return RR, RR_l, RR_u

    def get_graph_sig(self, RR, RR_l, RR_u):
        RR_dist1 = np.copy(RR)
        RR_dist2 = np.copy(RR)
        is_sig = (RR_l > 1) | (RR_u < 1)
        RR_dist1[~is_sig] = 1
        RR_graph = RR_dist1 - np.diag(np.diagonal(RR_dist1))
        RR_dist = RR_dist2.ravel()
        return RR_graph, RR_dist

    def calculate_phi(self, CC, N, net_type):
        Phi, t = self.get_phi(CC, N)
        Phi_graph, Phi_dist = self.get_graph_phi(Phi, t)
        if net_type != 'all':
            if net_type == 'high':
                Phi_graph[Phi_graph <= 0] = 0
                Phi_dist = Phi_dist[Phi_dist > 0]
            if net_type == 'less':
                Phi_graph[Phi_graph >= 0] = 0
                Phi_graph = Phi_graph * -1
                Phi_dist = Phi_dist[Phi_dist < 0]
        return Phi_dist, Phi_graph

    def get_phi(self, CC, N):
        P = np.diagonal(CC.toarray())
        PP = self.product_matrix(P)
        NP = self.product_matrix(N - P)

        Phi_num = N * CC.toarray() - PP
        Phi_dem = np.sqrt(PP * NP)
        Phi = Phi_num / Phi_dem

        sample_size = self.get_coprevalence(P)

        t_num = Phi * np.sqrt(sample_size - 2)
        t_den = np.sqrt(1 - (Phi ** 2))
        t = t_num / t_den

        Phi[~np.isfinite(Phi)] = 0
        t[~np.isfinite(t)] = 0
        return Phi, t

    def get_graph_phi(self, Phi, t):
        Phi_dist1 = np.copy(Phi)
        Phi_dist2 = np.copy(Phi)
        is_sig = (t <= -1.96) | (t >= 1.96)
        Phi_dist1[~is_sig] = 0
        Phi_graph = Phi_dist1 - np.diag(np.diagonal(Phi_dist1))
        Phi_dist = Phi_dist2.ravel()
        return Phi_graph, Phi_dist


if __name__ == '__main__':
    # Setting up paths and parameters
    data_path = os.path.abspath(os.path.join(PROJECT_ROOT, '../../..', 'pcpe'))
    # year = '2020' # or 'all', '' for all years
    typologies = ['I-d', 'I-e', 'IV-n']
    min_subjects = 5
    min_occurrences = 2
    net_type = 'high' # or 'less', 'all' depending on what type of network you want
    years = ['all'] + [str(year) for year in range(2019,2023)]
    data_ = DataPreparer.read_data(data_path)
    data_final_typology = DataPreparer.read_final_typology_label(data_path)
    for year in tqdm(years, desc='Creating Graphs Years', unit='Year', position=0):
        output_dir = os.path.join(PROJECT_ROOT, 'outputs')
        output_path = os.path.join(output_dir, f'{year}_{net_type}_{min_subjects}_{min_occurrences}')

        # print("Preparing data...")
        contas, sparse_matrix = DataPreparer.prepare_dataframe_for_cooccurrence(data_, year, typologies)

        df_quant, df_par_quant = DataPreparer.create_dataframes_for_attr(data_, year, typologies)

        # Ensuring the output directory exists
        FileManager.create_dir(output_dir)
        FileManager.create_dir(output_path)

        # print("Generating co-occurrence network...")
        graph_strategy = DefaultGraphStrategy()

        network_generator = NetworkCoOccurrence()

        # Getting the network
        C, CC, RR_graph, RR_dist, Phi_graph, Phi_dist, P, L = network_generator.get_network(
            labels=contas,
            occurrences=sparse_matrix,
            min_subjects_=min_subjects,
            min_occurrences_=min_occurrences,
            net_type_=net_type
        )

        # node_att, edge_att = DataPreparer.create_attr_for_nodes_edges(RR_graph, P, L, df_quant, df_par_quant,
        #                                                               typologies)
        #
        # G_rr = graph_strategy.create_graph(RR_graph, node_att, edge_att)

        node_att, edge_att = DataPreparer.create_attr_for_nodes_edges(Phi_graph, P, L, df_quant, df_par_quant,
                                                                      typologies, data_final_typology, data_)

        G_phi = graph_strategy.create_graph(Phi_graph, node_att, edge_att)

        FileManager.save_arrays_to_zip_as_csv([C.toarray(), CC.toarray(), RR_graph, RR_dist, Phi_graph, Phi_dist],
                                              ['C', 'CC', 'RR_graph', 'RR_dist', 'Phi_graph', 'Phi_dist'],
                                              output_path)
        # FileManager.write_gml(G_rr, os.path.join(output_path, 'network_graph_rr.gml'))
        FileManager.write_gml(G_phi, os.path.join(output_path, 'network_graph_phi.gml'))

        # print("Network saved successfully.")
