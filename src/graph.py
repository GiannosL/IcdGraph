from collections import defaultdict

import networkx as nx
from typing import List, Dict, Tuple

from src import Parameters
from src.hpo import HPOtoPheCode
from src.chapter_data import ICD
from src.phecodes import PheCodes


class Graph:
    def __init__(
            self,
            p: Parameters,
            icd_data: ICD,
            phecode_data: PheCodes,
            hpo_data: HPOtoPheCode
    ):
        # edge dictionary
        self.edge_dictionary = {}

        # ICD-10 hierarchical dictionary
        self.icd = icd_data.hierarchical_dictionary
        self.edge_dictionary[('icd', 'decomposes_to', 'icd')] = self.get_icd_hierarchical_edge_list()

        # PheCode to ICD-10 edge-list
        self.phecode = phecode_data
        self.edge_dictionary[('phecode', 'is_equivalent_to', 'icd')] = self.get_phecode2icd_edge_list()

        # HPO to PheCode
        self.hpo = hpo_data
        self.edge_dictionary[('hpo', 'is_connected_to', 'phecode')] = self.get_hpo2phecode_edge_list()

        # gather information on data
        self.node_dictionary = self.get_info()

        # once all data has been pre-processed generate graphs
        self.networkx_graph = self.get_networkx_graph(params=p)

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


