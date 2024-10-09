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
    def save_graphml(graph, path):
        nx.write_graphml(graph, path)

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
    """Handles data preparation for network analysis."""
    @staticmethod
    def prepare_dataframe(data_path, year=None):
        year = '' if year is None or year in ['', 'all'] else year
        dtype_dict = {
            'NUMERO_CASO': 'str',
            'NUMERO_BANCO': 'str',
            'NOME_BANCO': 'str',
            'NUMERO_AGENCIA': 'str',
            'NUMERO_CONTA': 'str',
            'TIPO': 'str',
            'NOME_TITULAR': 'str',
            'CPF_CNPJ_TITULAR': 'str',
            'DESCRICAO_LANCAMENTO': 'str',
            'CNAB': 'str',
            'DATA_LANCAMENTO': 'str',
            'LOCAL_TRANSACAO': 'str',
            'VALOR_TRANSACAO': 'float64',
            'NATUREZA_LANCAMENTO': 'str',
            'VALOR_SALDO': 'float64',
            'NATUREZA_SALDO': 'str',
            'CPF_CNPJ_OD': 'str',
            'NOME_PESSOA_OD': 'str',
            'TIPO_PESSOA_OD': 'str',
            'NUMERO_BANCO_OD': 'str',
            'NUMERO_AGENCIA_OD': 'str',
            'NUMERO_CONTA_OD': 'str',
            'I-d': 'str',
            'DIA_LANCAMENTO': 'str',
            'MES_LANCAMENTO': 'str',
            'ANO_LANCAMENTO': 'str'
        }

        df = pd.read_csv(os.path.join(data_path, 'pcpe_02.csv'), sep=';', decimal=',', dtype=dtype_dict)
        df['I-d'] = df['I-d'].fillna('0')
        if year:
            df = df.loc[df['ANO_LANCAMENTO'] == year]

        df['CONTA_ORIGEM'] = df['NUMERO_AGENCIA'] + '_' + df['NUMERO_CONTA']
        df['CONTA_DESTINO'] = df['NUMERO_AGENCIA_OD'] + '_' + df['NUMERO_CONTA_OD']
        df['CONTA_DESTINO'] = df['CONTA_DESTINO'].fillna('REMOVE')
        df.loc[df['CONTA_DESTINO'] == '0_0', 'CONTA_DESTINO'] = 'REMOVE'

        df = df.reset_index(drop=True)

        contas_origem = pd.Index(df['CONTA_ORIGEM'].unique())
        contas_destino = pd.Index(df['CONTA_DESTINO'].unique())
        contas = np.concatenate((contas_origem, contas_destino[~np.isin(contas_destino, contas_origem)])).tolist()

        df_origem = df[['CONTA_ORIGEM', 'I-d']].rename(columns={'CONTA_ORIGEM': 'CONTA'})
        df_destino = df[['CONTA_DESTINO', 'I-d']].rename(columns={'CONTA_DESTINO': 'CONTA'})
        df_conta_id = pd.concat([df_origem, df_destino])

        df_count = df_conta_id.groupby(['CONTA', 'I-d']).size().unstack(fill_value=0).reset_index()
        if '1' not in df_count.columns:
            df_count['1'] = 0

        df_pair_count = df.groupby(['CONTA_ORIGEM', 'CONTA_DESTINO', 'I-d']).size().unstack(fill_value=0).reset_index()
        if '1' not in df_pair_count.columns:
            df_pair_count['1'] = 0

        return contas, df_count, df_pair_count, DataPreparer.create_sparse_matrix(df, contas)

    @staticmethod
    def create_sparse_matrix(df, contas):
        contas_dict = {conta: idx for idx, conta in enumerate(contas)}
        origem_factors = df['CONTA_ORIGEM'].map(contas_dict).values
        destino_factors = df['CONTA_DESTINO'].map(contas_dict).values

        n_rows = df.shape[0]
        n_cols = len(contas)

        row_indices = np.concatenate([np.arange(n_rows), np.arange(n_rows)])
        col_indices = np.concatenate([origem_factors, destino_factors])
        data = np.ones(len(row_indices))
        data[len(origem_factors):][destino_factors == contas_dict['REMOVE']] = 0

        C_sparse = coo_matrix((data, (row_indices, col_indices)), shape=(n_rows, n_cols)).tocsr()

        return C_sparse


class IGraphStrategy(ABC):
    """Interface Segregation: Abstract class for graph generation strategies."""
    @abstractmethod
    def create_graph(self, matrix, P, L, df_count, df_pair_count):
        pass


class DefaultGraphStrategy(IGraphStrategy):
    """Concrete implementation of graph generation."""
    def create_graph(self, matrix, P, L, df_count, df_pair_count):
        G = nx.from_numpy_array(matrix)
        self.put_attributes(G, P, L, matrix, df_count, df_pair_count)
        return G

    def put_attributes(self, G, P, L, metric, df_count, df_pair_count):
        df_count = df_count[df_count['CONTA'].isin(L)]
        # filtered_df_pair_count = df_pair_count[
        #     df_pair_count['CONTA_ORIGEM'].isin(L) | df_pair_count['CONTA_DESTINO'].isin(L)
        # ]

        metric_sums = np.sum(metric, axis=1)

        node_attrs = {
            i: {
                'prevalence': int(P[i]),
                'sum_metric': float(metric_sums[i]),
                'label': L[i],
                'quantity_i-d': int(df_count.loc[df_count['CONTA'] == L[i], '1'].values[0])
                if not df_count[df_count['CONTA'] == L[i]].empty else 0
            }
            for i in G.nodes
        }
        nx.set_node_attributes(G, node_attrs)

        edge_attrs = {}
        for _, row in df_pair_count.iterrows():
            conta_origem = row['CONTA_ORIGEM']
            conta_destino = row['CONTA_DESTINO']
            valor_coluna_1 = row['1']

            if conta_origem in L and conta_destino in L:
                idx_origem = L.index(conta_origem)
                idx_destino = L.index(conta_destino)
                 
                edge_attrs[(idx_origem, idx_destino)] = {'transacoes_Id_entre_contas': valor_coluna_1}
        nx.set_edge_attributes(G, edge_attrs)


class DefaultGraphRandomStrategy(IGraphStrategy):
    """Concrete implementation of graph generation."""
    def create_graph(self, matrix, P, L, df_count, df_pair_count):
        G = nx.from_numpy_array(matrix)
        self.put_attributes(G, P, L, matrix, df_count, df_pair_count)
        return G

    def put_attributes(self, G, P, L, metric, df_count, df_pair_count):
        df_count = df_count[df_count['CONTA'].isin(L)]
        df_count = self.randomize_attribute(df_count, '1')
        # filtered_df_pair_count = df_pair_count[
        #     df_pair_count['CONTA_ORIGEM'].isin(L) | df_pair_count['CONTA_DESTINO'].isin(L)
        # ]

        metric_sums = np.sum(metric, axis=1)

        node_attrs = {
            i: {
                'prevalence': int(P[i]),
                'sum_metric': float(metric_sums[i]),
                'label': L[i],
                'quantity_i-d': int(df_count.loc[df_count['CONTA'] == L[i], '1'].values[0])
                if not df_count[df_count['CONTA'] == L[i]].empty else 0
            }
            for i in G.nodes
        }
        nx.set_node_attributes(G, node_attrs)

        edge_attrs = {}
        for _, row in df_pair_count.iterrows():
            conta_origem = row['CONTA_ORIGEM']
            conta_destino = row['CONTA_DESTINO']
            valor_coluna_1 = row['1']

            if conta_origem in L and conta_destino in L:
                idx_origem = L.index(conta_origem)
                idx_destino = L.index(conta_destino)
                 
                edge_attrs[(idx_origem, idx_destino)] = {'transacoes_Id_entre_contas': valor_coluna_1}
        nx.set_edge_attributes(G, edge_attrs)

    @staticmethod
    def randomize_attribute(df, column_name):
        column_values = df[column_name].values

        num_zeros = np.sum(column_values == 0)
        num_non_zeros = len(column_values) - num_zeros

        new_non_zero_values = np.random.randint(1, np.max(column_values) + 1, size=num_non_zeros)
        randomized_values = np.concatenate([np.zeros(num_zeros, dtype=column_values.dtype), new_non_zero_values])
        np.random.shuffle(randomized_values)

        df.loc[:, column_name] = randomized_values
        return df


class NetworkCoOccurrence:
    """Handles co-occurrence network generation using strategies."""
    def __init__(self, graph_strategy: IGraphStrategy):
        self.graph_strategy = graph_strategy

    def get_network(self, labels, occurrences, df_count, df_pair_count, min_subjects=5, min_occurrences=2, net_type=None):
        C, CC, L = self.get_cooccurrence(occurrences.copy(), labels, min_subjects, min_occurrences)
        N = C.shape[0]

        RR_dist, RR_graph = self.calculate_risk_ratio(CC, N, net_type)

        P = np.diag(CC.toarray())
        G_rr = self.graph_strategy.create_graph(RR_graph, P, L, df_count, df_pair_count)

        Phi_dist, Phi_graph = self.calculate_phi(CC, N, net_type)

        G_phi = self.graph_strategy.create_graph(Phi_graph, P, L, df_count, df_pair_count)

        return C, CC, RR_graph, RR_dist, G_rr, Phi_graph, Phi_dist, G_phi, P, L

    def get_cooccurrence(self, occurrence, L, min_subjects=5, min_occurrences=2):
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
    random_i_d = True
    min_subjects = 5
    min_occurrences = 2
    net_types = ['high', 'less', 'all']  # or 'less', 'all' depending on what type of network you want
    years = [str(year) for year in range(2015,2023)] #+ ['all']
    for net_type in tqdm(net_types, desc='Creating Graphs', unit='Net_Type'):
        for year in tqdm(years, desc='Creating Graphs', unit='Year'):
            output_path = os.path.join(
                PROJECT_ROOT, f'{net_type}_{min_subjects}_{min_occurrences}_{year}{"_random1" if random_i_d else ""}')

            # print("Preparing data...")
            contas, df_count, df_pair_count, sparse_matrix = DataPreparer.prepare_dataframe(data_path, year)

            if random_i_d and (df_count['1'] == 0).all():
                continue

            # Ensuring the output directory exists
            FileManager.create_dir(output_path)

            # print("Generating co-occurrence network...")
            if random_i_d:
                graph_strategy = DefaultGraphRandomStrategy()
            else:
                graph_strategy = DefaultGraphStrategy()

            network_generator = NetworkCoOccurrence(graph_strategy)

            # Getting the network
            C, CC, RR_graph, RR_dist, G_rr, Phi_graph, Phi_dist, G_phi, P, L = network_generator.get_network(
                labels=contas,
                occurrences=sparse_matrix,
                df_count=df_count,
                df_pair_count=df_pair_count,
                min_subjects=min_subjects,
                min_occurrences=min_occurrences,
                net_type=net_type
            )

            FileManager.save_arrays_to_zip_as_csv([C.toarray(), CC.toarray(), RR_graph, RR_dist, Phi_graph, Phi_dist],
                                                  ['C', 'CC', 'RR_graph', 'RR_dist', 'Phi_graph', 'Phi_dist'],
                                                  output_path)
            FileManager.save_graphml(G_rr, os.path.join(output_path, 'network_graph_rr.graphml'))
            FileManager.save_graphml(G_phi, os.path.join(output_path, 'network_graph_phi.graphml'))
            # print("Network saved successfully.")
