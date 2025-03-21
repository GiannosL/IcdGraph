import pandas as pd
from typing import Tuple


class HPOtoPheCode:
    def __init__(self):
        self.phecode_list = []
        self.hpo_list = []

        # unique values
        self.unique_phecodes = []
        self.unique_hpos = []

        # entity info
        self.phecode_info = {}
        self.hpo_info = {}

    def add(
            self,
            phecode: str,
            phecode_label: str,
            phecode_category: str,
            hpo: str,
            hpo_label: str
    ):
        self.phecode_list.append(phecode)
        self.hpo_list.append(hpo)

        self.phecode_info[phecode] = {
            'name': phecode_label,
            'category': phecode_category,
        }

        self.hpo_info[hpo] = hpo_label

    def process(self):
        """
        Run once the dataset has been loaded through `add`.
        """
        self.unique_phecodes = sorted(set(self.phecode_list))
        self.unique_hpos = sorted(set(self.hpo_list))

    def __len__(self) -> int:
        return len(self.phecode_list)

    def __getitem__(self, i: int) -> Tuple[str, str]:
        return self.hpo_list[i], self.phecode_list[i]


def parse_hpos(file_name: str) -> HPOtoPheCode:
    """
    Parses input file and generates HPO object.

    `file_name`: input file string, tab-seperated file
    """
    # parse file
    cols = [
        'phecode1.2_code',
        'phecode1.2_label',
        'phecode1.2_category',
        'hpo_code',
        'hpo_label',
            ]
    df = pd.read_csv(file_name, sep='\t', usecols=cols, dtype=str)
    df = df.dropna()

    # save to HPO object
    hpo = HPOtoPheCode()
    for phecode, phe_lab, phe_cat, hpo_code, hpo_lab in df.to_records(index=False):
        hpo.add(
            phecode=phecode,
            phecode_label=phe_lab,
            phecode_category=phe_cat,
            hpo=hpo_code,
            hpo_label=hpo_lab
        )

    hpo.process()

    return hpo
