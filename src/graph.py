import torch
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import List, Dict, Tuple
from torch_geometric.data import HeteroData

from src import Parameters
from src.hpo import HPOtoPheCode
from src.chapter_data import ICD
from src.phecodes import PheCodes
from src.primekg import PrimeKG


class Graph:
    def __init__(
            self,
            p: Parameters,
            icd_data: ICD,
            phecode_data: PheCodes,
            hpo_data: HPOtoPheCode,
            primekg_data: PrimeKG
    ):
        # edge dictionary
        self.edge_dictionary = {}

        # ICD-10 hierarchical dictionary
        self.icd = icd_data.hierarchical_dictionary
        self.edge_dictionary[('icd', 'decomposes_to', 'icd')] = self.get_icd_hierarchical_edge_list()
        self.edge_dictionary[('icd', 'is_part_of', 'icd')] = get_reverse_edgelist(
            edge_list=self.edge_dictionary[('icd', 'decomposes_to', 'icd')]
        )

        # PheCode to ICD-10 edge-list
        self.phecode = phecode_data
        self.edge_dictionary[('phecode', 'is_equivalent_to', 'icd')] = self.get_phecode2icd_edge_list()
        self.edge_dictionary[('icd', 'is_also_eq_to', 'phecode')] = get_reverse_edgelist(
            edge_list=self.edge_dictionary[('phecode', 'is_equivalent_to', 'icd')]
        )

        # HPO to PheCode
        self.hpo = hpo_data
        self.edge_dictionary[('hpo', 'is_connected_to', 'phecode')] = self.get_hpo2phecode_edge_list()
        self.edge_dictionary[('phecode', 'rev_connection_to', 'hpo')] = get_reverse_edgelist(
            edge_list=self.edge_dictionary[('hpo', 'is_connected_to', 'phecode')]
        )

        # Integrate PrimeKG
        self.primekg = primekg_data.edge_dictionary
        for edge_type, edge_list in self.primekg.items():
            self.edge_dictionary[edge_type] = edge_list

        # gather information on data
        self.node_dictionary = self.get_info()

        # once all data has been pre-processed generate graphs
        self.node_encryptions = {}
        self.networkx_graph = self.get_networkx_graph(params=p)
        self.torch_graph = self.get_torch_graph()

        self.rev_node_encryptions = {}
        for node_type, my_node_dict in self.node_encryptions.items():
             self.rev_node_encryptions[node_type] = {v: k for k, v in my_node_dict.items()}

    def get_icd_hierarchical_edge_list(self) -> List[Tuple[str, str]]:
        """
        Generate an edge-list for ICD-10 codes.
        `returns` List of edges (tuples with two strings).
        """
        edge_list = []
        for upper_code, lower_code_list in self.icd.items():
            for lower_code in lower_code_list:
                edge_list.append(
                    (upper_code, lower_code)
                )

        return edge_list

    def get_phecode2icd_edge_list(self) -> List[Tuple[str, str]]:
        """
        Generate an edge-list from PheCodes to ICD-10 codes.
        `returns` List of edges (PheCode, ICD code)
        """
        edge_list = []
        for i in range(len(self.phecode)):
            edge_list.append(
                self.phecode[i]
            )

        return edge_list

    def get_hpo2phecode_edge_list(self) -> List[Tuple[str, str]]:
        """
        Generate an edge-list from HPO to PheCodes.
        `returns` List of edges (HPO, PheCode)
        """
        edge_list = []
        for i in range(len(self.hpo)):
            edge_list.append(
                self.hpo[i]
            )

        return edge_list

    def get_networkx_graph(self, params: Parameters) -> nx.DiGraph:
        """
        Generates a networkx graph from the edge dictionary.
        """
        # collect nodes and node-features
        node_features = {}
        for edge_type, edge_list in self.edge_dictionary.items():
            for node_a, node_b in edge_list:
                node_features[node_a] = {
                    'type': edge_type[0],
                    'color': params.get_color(node_type=edge_type[0])
                }
                node_features[node_b] = {
                    'type': edge_type[2],
                    'color': params.get_color(node_type=edge_type[2])
                }

        # empty directed graph object
        my_graph = nx.DiGraph()

        for node, feats in node_features.items():
            my_graph.add_node(node, **feats)

        for edge_type, edge_list in self.edge_dictionary.items():
            my_graph.add_edges_from(edge_list)

        # get node degrees
        node_degrees = dict(my_graph.degree())
        nx.set_node_attributes(my_graph, node_degrees, name='degree')

        return my_graph

    def get_info(self) -> Dict[str, List[str]]:
        """
        Gathers information on the dataset
        """
        nodes = defaultdict(list)
        for edge_type, edge_list in self.edge_dictionary.items():
            for my_edge in edge_list:
                nodes[edge_type[0]].append(my_edge[0])
                nodes[edge_type[2]].append(my_edge[1])
        nodes = dict(nodes)

        for node_type, node_list in nodes.items():
            nodes[node_type] = sorted(set(node_list))

        return nodes

    def graph_statistics(self, outfile: str = ''):
        """
        `outfile`: path to output file to save statistics inside. If left empty will print to console.
        """
        #
        my_line = ''

        # get node stats
        for node_type, node_list in self.node_dictionary.items():
            my_line += f'{node_type}\t{len(node_list):,}\n'

        # get edge stats
        for edge_type, edge_list in self.edge_dictionary.items():
            my_line += f'{" ".join(edge_type)}\t{len(edge_list):,}\n'

        if outfile:
            with open(outfile, 'w') as f:
                f.write(my_line)
        else:
            print(my_line)

    def get_torch_graph(self) -> HeteroData:
        """
        Generates torch `HeteroData` graph object.
        """
        #
        # get nodes and encode them
        node_dict = defaultdict(list)
        for edge_type, edge_list in self.edge_dictionary.items():
            for node_a, node_b in edge_list:
                node_dict[edge_type[0]].append(node_a)
                node_dict[edge_type[2]].append(node_b)
        node_dict = dict(node_dict)
        for node_type, node_list in node_dict.items():
            sorted_nodes = sorted(set(node_list))
            self.node_encryptions[node_type] = {node: i for i, node in enumerate(sorted_nodes)}

        edge_dict = {}
        for edge_type, edge_list in self.edge_dictionary.items():
            edge_dict[edge_type] = []
            for node_a, node_b in edge_list:
                edge_dict[edge_type].append(
                    (self.node_encryptions[edge_type[0]][node_a], self.node_encryptions[edge_type[2]][node_b])
                )

        # generate heterogeneous graph
        my_graph = HeteroData()
        # nodes
        for node_type, node_dict in self.node_encryptions.items():
            my_graph[node_type].node_ids = torch.arange(len(node_dict.keys())).to(torch.int64)
            # TODO: replace with actual feautres
            my_graph[node_type].x = torch.ones((len(node_dict.keys()), 7)).to(torch.float32)

        for edge_type, edge_list in edge_dict.items():
            edge_array = np.array(edge_list).T
            my_graph[edge_type].edge_index = torch.from_numpy(edge_array).to(torch.int64)

        return my_graph


def get_reverse_edgelist(edge_list: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Reverses the imput edgelist.
    `params: edge_list` List containing tuple of edges.
    `return` List of edges reversed
    """

    return [(edge[1], edge[0]) for edge in edge_list]
