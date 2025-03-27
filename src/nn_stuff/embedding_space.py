import torch
from typing import List, Dict


class EmbeddingSpace:
    def __init__(self, embed_dict: Dict[str, torch.Tensor]):
        self.node_type_list = sorted(embed_dict.keys())
        self.embedding_dictionary = embed_dict

    def node_types(self) -> List[str]:
        return self.node_type_list

    def __call__(self) -> Dict[str, torch.Tensor]:
        return self.embedding_dictionary

    def __getitem__(self, node_type: str) -> torch.Tensor:
        return self.embedding_dictionary[node_type].detach().cpu()
