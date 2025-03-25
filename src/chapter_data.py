from typing import List, Tuple
from collections import defaultdict


class ICD:
    def __init__(self):
        self.codes = []
        self.descriptions = []
        self.hierarchical_dictionary = {}
        self.level_1_chapters = []
    
    def add(self, code: str, desc: str):
        self.codes.append(code)
        self.descriptions.append(desc)

    def process(self):
        """
        Collects information on the dataset.
        """
        # create a dictionary with a hierarchical representation of the ICD codes
        self.get_code_hierarchies()

        # get level 1 chapters in the dataset
        self.get_highest_chapters()

    def get_code_hierarchies(self):
        """
        Iterate over codes in the existing dataset to create a dictionary
        with keys being upper level chapters (e.g. A00) and the values being
        lists of lower level chapters (e.g. A00.0, A00.1).
        """
        # collect different kinds of lengths
        count_hierarchy = defaultdict(list)
        for my_code in self.codes:
            count_hierarchy[len(my_code)].append(my_code)

        code_hierarchy = defaultdict(list)
        code_lengths = sorted(count_hierarchy.keys())
        # iterate over all code lengths apart from the last (because it's at the hierarchy bottom)
        for i in range(len(code_lengths)-1):
            # get the upper code (e.g. A00)
            for upper_code in count_hierarchy[code_lengths[i]]:
                # collect the lower codes for it (e.g. A00.0, A00.1)
                for lower_code in count_hierarchy[code_lengths[i+1]]:
                    if lower_code.startswith(upper_code):
                        code_hierarchy[upper_code].append(lower_code)

        self.hierarchical_dictionary = dict(code_hierarchy)

    def get_highest_chapters(self):
        """
        Collect the highest level ICD-10 codes in the dataset.
        e.g. A, B, C, ...
        """
        level_1_chapters = []
        for key in self.hierarchical_dictionary.keys():
            level_1_chapters.append(key[0])
        self.level_1_chapters = sorted(set(level_1_chapters))

    def __getitem__(self, i: int) -> Tuple[str, str]:
        return self.codes[i], self.descriptions[i]
    
    def __call__(self) -> List[str]:
        return self.codes
    
    def __len__(self) -> int:
        return len(self.codes)


def parse_icd(file_list: List[str]) -> ICD:
    """
    `filename`: path to the ICD-10 input file
    `return`: ICD object containing all codes in the file
    """
    # empty ICD object
    icd_codes = ICD()

    # iterate over files
    for file_name in file_list:
        with open(file_name, 'r') as f:
            contents = f.read().splitlines()

        for line in contents[1:]:
            split_line = line.split(';')
            icd_codes.add(
                code=split_line[0],
                desc=split_line[1]
            )

    # apply preprocessing to the finished dataset
    icd_codes.process()

    return icd_codes
