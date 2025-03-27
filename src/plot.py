import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Dict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src import Parameters
from src.graph import Graph
from src.nn_stuff.embedding_space import EmbeddingSpace


class Plot:
    def __init__(self):
        self.figsize = (14, 8)
        self.tl = 20
        self.xyl = 16
        self.dpi = 800

    def graph(self, g: Graph, outfile: str = ''):
        """
        Simple plot networkx graph.
        """
        g = g.networkx_graph

        # plotting
        plt.figure(figsize=self.figsize)

        nx.draw(
            g,
            with_labels=True,
            edge_color='#B09E99',
            font_size=10,
            node_color=[g.nodes()[n]['color'] for n in g.nodes()],
        )

        if outfile:
            plt.savefig(outfile, dpi=self.dpi)
            plt.close()
        else:
            plt.show()

    def interactive_graph(self, g: Graph, outfile: str = '', layout: str = 'spring'):
        """
        Generate interactive graph.
        """
        # get node positions on the page
        g = g.networkx_graph
        if layout == 'spring':
            node_positions = nx.spring_layout(g)

        # set nodes
        nodes = get_node_traces(g=g, pos=node_positions)
        # set edges
        edges = _get_edge_traces(g=g, pos=node_positions)

        # plot traces
        node_scatter = go.Scatter(
            x=nodes.x,
            y=nodes.y,
            mode='markers',
            hoverinfo='text',
            marker={'color': nodes.colors, 'size': nodes.degrees},
            text=nodes.names,
        )

        edge_scatter = go.Scatter(
            x=edges.x,
            y=edges.y,
            mode='lines',
            line={'color':'#FFFDD0'},
            hoverinfo='text'
        )

        # Figure plot
        lay = go.Layout(
            hovermode='closest',
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            template='plotly_dark',
            height=800
        )

        fig = go.Figure(
            data=[edge_scatter, node_scatter],
            layout=lay
        )
        fig.update_layout(
            autosize=True,
            height=1440,  # Adjust height based on your screen
            width=2560,  # Adjust width based on your screen
        )

        if outfile:
            fig.write_html(outfile)
        else:
            fig.show()

    def embedding_space(
            self,
            parameters: Parameters,
            emb: EmbeddingSpace,
            title: str,
    ):
        """
        Plot embedding space.
        """
        embeddings, sizes, nodes = [], [], []
        for node_type in emb.node_types():
            embeddings.append(emb[node_type])
            sizes.append(emb[node_type].shape[0])
            nodes.append(node_type)
        # concatenate (stack) tensors
        embeddings = torch.cat(embeddings, dim=0)

        # perform PCA to reduce the dimensionality of the dataset
        pca_results = dimensionality_reduction(data=embeddings)

        # Compute cumulative sizes for correct indexing
        cumulative_sizes = [0] + torch.cumsum(torch.tensor(sizes), dim=0).tolist()
        colors = [parameters.get_color(node_type=nt) for nt in nodes]

        # plotting
        plt.figure(figsize=self.figsize)

        # plot scatters per node type
        for i, node in enumerate(nodes):
            start, end = cumulative_sizes[i], cumulative_sizes[i + 1]
            plt.scatter(
                pca_results.pcs[start:end, 0],
                pca_results.pcs[start:end, 1],
                c=colors[i],
                alpha=0.6,
                label=node
            )

        plt.title(title, fontsize=self.tl)
        plt.xlabel(f'PC-1, {pca_results.var[0]:.2}%', fontsize=self.xyl)
        plt.ylabel(f'PC-2, {pca_results.var[1]:.2}%', fontsize=self.xyl)
        plt.legend()

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)

        plt.xticks([])
        plt.yticks([])

        plt.show()


class NodeCoordinates:
    def __init__(self, x: List[int], y: List[int], names: List[str], colors: List[str], degrees: List[int]):
        self.x = x
        self.y = y
        self.names = names
        self.colors = colors
        self.degrees = degrees


class EdgeCoordinates:
    def __init__(self, x: List[int], y: List[int]):
        self.x = x
        self.y = y


class ResultsPCA:
    def __init__(self, principle_components: np.array, variance_ratio: np.array):
        self.pcs = principle_components
        self.var = variance_ratio * 100


def get_node_traces(g: Graph, pos: Dict) -> NodeCoordinates:
    """
    Set node positions.
    """
    nodes_x, nodes_y = [], []
    node_names, node_colors = [], []
    node_degrees = []

    for node, node_d in g.nodes(data=True):
        x, y = pos[node]
        nodes_x.append(x)
        nodes_y.append(y)
        node_names.append(f'Name: {node}<br>Category: {node_d["type"]}<br>Degree: {node_d["degree"]}')
        node_colors.append(node_d['color'])
        node_degrees.append(node_d['degree'])

    return NodeCoordinates(x=nodes_x, y=nodes_y, names=node_names, colors=node_colors, degrees=node_degrees)


def _get_edge_traces(g: Graph, pos: Dict):
    """
    Edge coordinates.
    """
    edges_x, edges_y = [], []

    for edge in g.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edges_x.append(x0)
        edges_x.append(x1)
        edges_x.append(None)
        edges_y.append(y0)
        edges_y.append(y1)
        edges_y.append(None)

    return EdgeCoordinates(x=edges_x, y=edges_y)


def dimensionality_reduction(data: torch.Tensor) -> ResultsPCA:
    """
    Perform PCA to reduce the input tensor to two dimensions.
    """
    # start by normalizing the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # perform PCA
    pca = PCA(n_components=data.shape[1])
    pcs = pca.fit_transform(scaled_data)

    variance_ratio = pca.explained_variance_ratio_

    return ResultsPCA(principle_components=pcs, variance_ratio=variance_ratio)
