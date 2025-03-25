import torch
import pickle
import pandas as pd

from typing import List, Dict
from src import EdgeType
from src.graph import Graph
from src.nn_stuff.embedding_space import EmbeddingSpace


class ResultsGNN:
    def __init__(self, edge_type: EdgeType):
        # initialization variables
        self.edge_type = edge_type
        # variables to set later
        self.n_epochs = None
        self.loss_history = None
        self.validation_loss_history = None
        self.training_embedding_space = None
        self.validation_embedding_space = None
        self.test_embedding_space = None
        # results
        self.training_results = None
        self.training_eli = None
        self.training_labels = None
        self.validation_results = None
        self.validation_eli = None
        self.validation_labels = None
        self.test_results = None
        self.test_eli = None
        self.test_labels = None
        # formatted results
        self.dataframe = None

    def set_loss_history(self, history: List[float]):
        self.n_epochs = [i for i in range(1, len(history) + 1)]
        self.loss_history = history

    def set_validation_loss_history(self, history: List[float]):
        self.validation_loss_history = history

    def set_training_embeds(self, embs: EmbeddingSpace):
        self.training_embedding_space = embs

    def set_validation_embeds(self, embs: EmbeddingSpace):
        self.validation_embedding_space = embs

    def set_test_embeds(self, embs: EmbeddingSpace):
        self.test_embedding_space = embs

    def set_training_results(
            self,
            r: torch.Tensor,
            edge_label_index: torch.Tensor,
            edge_label: torch.Tensor
    ):
        self.training_results = r.tolist()
        self.training_eli = edge_label_index.tolist()
        self.training_labels = edge_label.tolist()

    def set_val_results(
            self,
            r: torch.Tensor,
            edge_label_index: torch.Tensor,
            edge_label: torch.Tensor
    ):
        self.validation_results = r.tolist()
        self.validation_eli = edge_label_index.tolist()
        self.validation_labels = edge_label.tolist()

    def set_test_results(
            self,
            r: torch.Tensor,
            edge_label_index: torch.Tensor,
            edge_label: torch.Tensor
    ):
        self.test_results = r.tolist()
        self.test_eli = edge_label_index.tolist()
        self.test_labels = edge_label.tolist()

    def annotate_results(self, kg: Graph):
        """
        Annotates the training, validation, and test results using the ID map from the knowledge graph.
        :param kg: KnowledgeGraph object, used for the kg.node_names_encoding attribute
        """

        def set_results(
                eli: List[List[int]],
                name_dict: Dict[str, Dict[int, str]],
                scores: List[float],
                labels: List[int],
                set_name: str) -> Dict[str, List[str | float]]:
            """
            Generate a dictionary with the two edge types annotated and a third key/value with the corresponding score.
            :param eli: List with two lists inside, representing the edge label index
            :param name_dict: Deconding dictionary from IDs (int) to the proper names (str)
            :param scores: The GNN output scores for the edges
            :param labels: List of true labels
            :param set_name: Name of the set train/validation/test
            :return:
            """
            res_dict = {self.edge_type[0]: [], self.edge_type[2]: [], 'true_label': [], 'score': [], 'set': []}

            node_names_a = name_dict[self.edge_type[0]]
            node_names_b = name_dict[self.edge_type[2]]

            # iterate over every edge and annotate the nodes and add the score
            for i in range(len(eli[0])):
                res_dict[self.edge_type[0]].append(
                    node_names_a[eli[0][i]]
                )
                res_dict[self.edge_type[2]].append(
                    node_names_b[eli[1][i]]
                )
                res_dict['true_label'].append(
                    labels[i]
                )
                res_dict['score'].append(
                    scores[i]
                )
                res_dict['set'].append(
                    set_name
                )

            return res_dict

        #
        train_annotated = set_results(eli=self.training_eli,
                                      name_dict=kg.rev_node_encryptions,
                                      scores=self.training_results,
                                      labels=self.training_labels,
                                      set_name='training')
        validation_annotated = set_results(eli=self.validation_eli,
                                           name_dict=kg.rev_node_encryptions,
                                           scores=self.validation_results,
                                           labels=self.validation_labels,
                                           set_name='validation')
        test_annotated = set_results(eli=self.test_eli,
                                     name_dict=kg.rev_node_encryptions,
                                     scores=self.test_results,
                                     labels=self.test_labels,
                                     set_name='test')

        # reset annotated results
        df_list = [pd.DataFrame(train_annotated), pd.DataFrame(validation_annotated), pd.DataFrame(test_annotated)]
        self.dataframe = pd.concat(df_list, ignore_index=True)

    def save_results(self, output_file: str):
        self.dataframe.to_csv(output_file, sep='\t', index=False)

    def save_loss_history(self, output_file: str):
        # combine loss histories
        loss_dict = {
            'training': self.loss_history,
            'validation': self.validation_loss_history,
        }
        with open(output_file, 'wb') as f:
            pickle.dump(loss_dict, f)

    def save_embeddings(
            self,
            training_outfile: str = '',
            validation_outfile: str = '',
            test_outfile: str = ''
    ):
        if training_outfile:
            with open(training_outfile, 'wb') as f:
                pickle.dump(self.training_embedding_space, f)

        if validation_outfile:
            with open(validation_outfile, 'wb') as f:
                pickle.dump(self.validation_embedding_space, f)

        if test_outfile:
            with open(test_outfile, 'wb') as f:
                pickle.dump(self.test_embedding_space, f)
