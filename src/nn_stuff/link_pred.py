import tqdm
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import GAT, to_hetero
from torch_geometric.transforms import RandomLinkSplit

from typing import Tuple
from src import EdgeType
from src.nn_stuff.embedding_space import EmbeddingSpace


class LinkPredictor(torch.nn.Module):
    def __init__(
            self,
            edge_type: EdgeType,
            train_graph: HeteroData,
            classifier_type: str,
            depth: int = 3,
    ):
        super().__init__()
        # make sure that classfier type is within selection
        classifier_types = ['linear', 'similarity']
        assert(classifier_type in classifier_types)

        # get edge type and size of input features
        self.edge_type = edge_type()
        in_features = train_graph[edge_type[0]].x.size(1)

        # initialize GNN encoder
        my_model = GAT(
            in_channels=in_features,
            hidden_channels=in_features*2,
            out_channels=16,
            num_layers=depth,
            #p=dropout,
            add_self_loops=False,
        )

        self.encoder = to_hetero(my_model, metadata=train_graph.metadata())

    def forward(self, graph: HeteroData) -> Tuple[torch.Tensor, EmbeddingSpace]:
        """
        Takes as input a graph and embeds the nodes.
        :param graph: a HeteroData object representing a graph
        :return: a tuple containing the results of prediction and the embedding space itself.
        """
        # generate node embeddings
        embeddings = self.encoder(
            x=graph.x_dict,
            edge_index=graph.edge_index_dict
        )

        # source and destination nodes
        src = graph[self.edge_type].edge_label_index[0]
        dst = graph[self.edge_type].edge_label_index[1]

        # similarity score
        predictions = (embeddings[self.edge_type[0]][src] @ embeddings[self.edge_type[2]][dst].T).sum(dim=-1)

        return predictions, EmbeddingSpace(embed_dict=embeddings)
