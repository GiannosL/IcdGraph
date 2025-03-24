import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Dict

from networkx.drawing import spring_layout

from src.graph import Graph


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
