class Parameters:
    def __init__(self):
        # local files
        self.icd_file = 'data/icd_chapter_i.txt'
        self.phecode_file = 'data/Phecode_map_v1_2_icd10_WHO_beta.csv'
        self.hpo_file = 'data/hpo-phecode1.2_links.tsv'

        # node colors
        self.icd_color = '#64B6AC'
        self.phecode_color = '#FAD4C0'
        self.hpo_color = '#C0FDFB'

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
