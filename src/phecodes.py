import pandas as pd
from typing import List, Tuple

class PheCodes:
    def __init__(self):
        # pairs
        self.icd10_list = []
        self.phecode_list = []

        # throwaway variable
        self.temp_phecode_names = []

        # unique versions
        self.icd10 = []
        self.phecodes = []
        self.phecode_names = {}

    def add(self, icd: str, phecode: str, phecode_desc: str):
        self.icd10_list.append(icd)
        self.phecode_list.append(phecode)
        self.temp_phecode_names.append(phecode_desc)

    def process(self):
        """
        Get unique ICD, PheCodes and create a map with phecode names.
        """
        self.icd10 = sorted((set(self.icd10_list)))
        self.phecodes = sorted((set(self.phecode_list)))

        # phecode name map
        for i in range(len(self.phecode_list)):
            self.phecode_names[self.phecode_list[i]] = self.temp_phecode_names[i]

    def __getitem__(self, i: int) -> Tuple[str, str]:
        return self.phecode_list[i], self.icd10_list[i]

    def __len__(self) -> int:
        return len(self.phecode_list)


def parse_phecodes(file_name: str) -> PheCodes:
    """
    Parses input file and generates PheCode object.

    `file_name`: input file string, comma-seperated file
    """
    # parse file
    cols = ['ICD10', 'PHECODE', 'Excl. Phenotypes']
    df = pd.read_csv(file_name, sep=',', usecols=cols, dtype=str)
    df = df.dropna()

    # save data into PheCodes object
    phecodes = PheCodes()
    for icd, phe, phe_desc in df.to_records(index=False):
        # TODO: remove cardiovascular filter
        if icd.startswith('I'):
            phecodes.add(
                icd=icd,
                phecode=phe,
                phecode_desc=phe_desc
            )

    return phecodes
