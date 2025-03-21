import networkx as nx
import matplotlib.pyplot as plt

from src.graph import Graph


class Plot:
    def __init__(self):
        self.figsize = (14, 8)
        self.tl = 20
        self.xyl = 16

    def graph(self, g: Graph):
        """
        Plot networkx graph
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

        plt.show()
