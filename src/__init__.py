from pathlib import Path
from typing import Tuple


class EdgeType:
    def __init__(self, src: str, rel: str, dst: str):
        self.edge_type = (src, rel, dst)

    def __call__(self) -> Tuple[str, str, str]:
        return self.edge_type

    def __getitem__(self, item: int) -> str:
        return self.edge_type[item]

    def __len__(self) -> int:
        return len(self.edge_type)


class Parameters:
    def __init__(self):
        # local files
        self.icd_files = [
            'data/icd_chapter_i.txt',
            'data/icd_chapter_j.txt',
            'data/icd_chapter_k.txt',
        ]
        self.phecode_file = 'data/Phecode_map_v1_2_icd10_WHO_beta.csv'
        self.hpo_file = 'data/hpo-phecode1.2_links.tsv'
        self.primekg_file = 'data/kg.csv'

        self.plot_file = Path('~/Desktop/plot.png').expanduser()
        self.interactive_plot_file = Path('~/Desktop/plot.html').expanduser()
        self.training_embedding_plot_file = Path('~/Desktop/training_embeddings.png').expanduser()

        # node colors
        self.icd_color = '#64B6AC'
        self.phecode_color = '#FAD4C0'
        self.hpo_color = '#C0FDFB'

        # edge of interest
        self.edge_type = EdgeType(
            src='phecode',
            rel='is_equivalent_to',
            dst='icd',
        )

        # GNN parameters
        self.epochs = 500
        self.learning_rate = 0.001

    def get_color(self, node_type: str) -> str:
        """
        `node_type`: node type within selected options
        `returns`: color hex code
        """
        if node_type == 'icd':
            return self.icd_color
        elif node_type == 'phecode':
            return self.phecode_color
        elif node_type == 'hpo':
            return self.hpo_color

        raise ValueError(f'Node type "{node_type}" not available!')
