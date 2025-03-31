import pandas as pd
from collections import defaultdict
from typing import List, Tuple
from src import EdgeType


class PrimeKG:
    def __init__(self):
        self.node_types = None
        self.edge_types = None
        self.edge_dictionary = defaultdict(list)
        self.node_dictionary = defaultdict(dict)

    def add(
            self,
            edge_type: str,
            node_id_a: str,
            node_a_type: str,
            node_a_name: str,
            node_id_b: str,
            node_b_type: str,
            node_b_name: str,
    ):
        my_edge = (node_a_type, edge_type, node_b_type)
        self.edge_dictionary[my_edge].append(
            (node_id_a, node_id_b)
        )
        # save node information
        self.node_dictionary[node_a_type][node_id_a] = node_a_name
        self.node_dictionary[node_b_type][node_id_b] = node_b_name

    def process(self):
        """
        Format inputs
        """
        new_edge_dict = {}
        for edge_type, edge_list in self.edge_dictionary.items():
            my_edge = [edge_type[0], edge_type[1], edge_type[2]]
            nodes_a = [n[0] for n in edge_list]
            nodes_b = [n[1] for n in edge_list]

            if edge_type[0] == 'gene/protein':
                my_edge[0] = 'protein'
            if edge_type[2] == 'gene/protein':
                my_edge[2] = 'protein'
            if edge_type[0] == 'effect/phenotype':
                my_edge[0] = 'hpo'
                nodes_a = ['HPO_' + f'{n:0>7}' for n in nodes_a]
            if edge_type[2] == 'effect/phenotype':
                my_edge[2] = 'hpo'
                nodes_b = ['HPO_' + f'{n:0>7}' for n in nodes_b]
            if edge_type[1] == 'off-label use':
                my_edge[1] = 'off_label'

            edge_list = [(nodes_a[i], nodes_b[i]) for i in range(len(nodes_a))]
            new_edge_dict[tuple(my_edge)] = edge_list
        self.edge_dictionary = new_edge_dict

        nd = {}
        for node_type, node_sub_dict in self.node_dictionary.items():
            if node_type == 'gene/protein':
                nd['protein'] = node_sub_dict
            elif node_type == 'effect/phenotype':
                nd['hpo'] = node_sub_dict
            else:
                nd[node_type] = node_sub_dict

        self.node_types = sorted(self.node_dictionary.keys())
        self.edge_types = sorted(self.edge_dictionary.keys())

    def __getitem__(self, edge_type: EdgeType) -> List[Tuple[str, str]]:
        return self.edge_dictionary[edge_type]


def parse_primekg(file_name: str, edge_types: List[str]) -> PrimeKG:
    """
    Parse primekg csv file.
    """
    # parse file
    df = pd.read_csv(file_name, sep=',', dtype=str)
    # keep only nodes of interest
    df = df[df.relation.isin(edge_types)]
    df = df.dropna().reset_index(drop=True)

    #
    data = PrimeKG()
    for _, row in df.iterrows():
        data.add(
            edge_type=row['relation'],
            node_id_a=row['x_id'],
            node_a_type=row['x_type'],
            node_a_name=row['x_name'],
            node_id_b=row['y_id'],
            node_b_type=row['y_type'],
            node_b_name=row['y_name'],
        )

    data.process()

    return data
