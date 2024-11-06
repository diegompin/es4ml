import networkx as nx
import os
from collections import defaultdict
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))




class TypologyMerger:
    def __init__(self, data_path, typology):
        self.data_path = data_path
        self.typology = typology


    

    def merge_typologies(self):
        networks = defaultdict(list)
        for typology in self.typology:
            network_path = os.path.join(self.data_path,typology)
            dirs = os.listdir(network_path)
            for dir in dirs:
                graph_path = os.path.join(network_path,dir)
                graph = nx.read_graphml(graph_path)
                networks[typology].append(graph)
        
        for i in range(5):
            graphs=[]
            for key in networks:
                graphs.append(networks[key][i])
            union = nx.compose(G = graphs[0],H=nx.compose(G=graphs[1],H=graphs[2]))
            quantity_id = [data.get('quantity_i-d') for node, data in union.nodes(data=True)]
            quantity_ie = [data.get('quantity_i-e') for node, data in union.nodes(data=True)]
            quantity_iv_n = [data.get('quantity_iv-n') for node, data in union.nodes(data=True)]
            envolvement_class = {
                'NO_ENVOLVEMENT': 0,
                'I-D ENVOLVEMENT': 1,
                'I-E ENVOLVEMENT': 2,
                'IV-N ENVOLVEMENT': 3,
                'I-D AND I-E ENVOLVEMENT': 4,
                'I-D AND IV-N ENVOLVEMENT': 5,
                'I-E AND IV-N ENVOLVEMENT': 6
            }
            node_attrs = {
            i: {
                'envolvement': quantity_id[int(i)] or quantity_ie[int(i)] or quantity_iv_n[int(i)],
                'envolvement_class':  envolvement_class['NO_ENVOLVEMENT'] if not quantity_id[int(i)] and not quantity_ie[int(i)] and not quantity_iv_n[int(i)] 
                else envolvement_class['I-D ENVOLVEMENT'] if quantity_id[int(i)] and not quantity_ie[int(i)] and not quantity_iv_n[int(i)] else
                envolvement_class['I-E ENVOLVEMENT'] if not quantity_id[int(i)] and  quantity_ie[int(i)] and not quantity_iv_n[int(i)] else
                envolvement_class['IV-N ENVOLVEMENT'] if not quantity_id[int(i)] and not quantity_ie[int(i)] and  quantity_iv_n[int(i)] else
                envolvement_class['I-D AND I-E ENVOLVEMENT'] if quantity_id[int(i)] and  quantity_ie[int(i)] and not quantity_iv_n[int(i)] else
                envolvement_class['I-D AND IV-N ENVOLVEMENT'] if quantity_id[int(i)] and not quantity_ie[int(i)] and  quantity_iv_n[int(i)] else
                envolvement_class['I-E AND IV-N ENVOLVEMENT'],
            }
            for i in union.nodes
        }
            nx.set_node_attributes(G=union, values=node_attrs)
            output_path = os.path.join(PROJECT_ROOT,"typology_network","UNION",f"union_{i+1}.graphml")
            nx.write_graphml(union,output_path)









if __name__ == '__main__':
    typologies = ['I-D', 'I-E','IV-N']
    data_path = os.path.join(PROJECT_ROOT,'typology_network')
    typ_merger = TypologyMerger(data_path=data_path, typology=typologies)
    typ_merger.merge_typologies()


