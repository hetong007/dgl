"""A mini synthetic dataset for graph classification benchmark."""
import math, os
import networkx as nx
import numpy as np

from .dgl_dataset import DGLDataset
from .utils import save_graphs, load_graphs, makedirs
from .. import backend as F
from ..convert import graph as dgl_graph
from ..transform import add_self_loop

__all__ = ['MiniGCDataset']

class MiniGCDataset(DGLDataset):
    """The dataset class.
    The datset contains 8 different types of graphs.
    * class 0 : cycle graph
    * class 1 : star graph
    * class 2 : wheel graph
    * class 3 : lollipop graph
    * class 4 : hypercube graph
    * class 5 : grid graph
    * class 6 : clique graph
    * class 7 : circular ladder graph
    .. note::
        This dataset class is compatible with pytorch's :class:`Dataset` class.
    Parameters
    ----------
    num_graphs: int
        Number of graphs in this dataset.
    min_num_v: int
        Minimum number of nodes for graphs
    max_num_v: int
        Maximum number of nodes for graphs
    seed : int, default is None
        Random seed for data generation
    """

    def __init__(self, num_graphs, min_num_v, max_num_v, seed=None,
                 save_graph=True, force_reload=False, verbose=False):
        self.num_graphs = num_graphs
        self.min_num_v = min_num_v
        self.max_num_v = max_num_v
        self.seed = seed
        self.save_graph = save_graph

        super(MiniGCDataset, self).__init__(name="minigc",
                                            force_reload=force_reload,
                                            verbose=verbose)

    def process(self):
        self.graphs = []
        self.labels = []
        self._generate(self.seed)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.
        Paramters
        ---------
        idx : int
            The sample index.
        Returns
        -------
        (dgl.Graph, int)
            The graph and its label.
        """
        return self.graphs[idx], self.labels[idx]

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        if os.path.exists(graph_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        if self.save_graph:
            graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
            save_graphs(str(graph_path), self.graphs, {'labels': self.labels})

    def load(self):
        graphs, label_dict = load_graphs(os.path.join(self.save_path, 'dgl_graph.bin'))
        self.graphs = graphs
        self.labels = label_dict['labels']

    @property
    def num_classes(self):
        """Number of classes."""
        return 8

    def _generate(self, seed):
        if seed is not None:
            np.random.seed(seed)
        self._gen_cycle(self.num_graphs // 8)
        self._gen_star(self.num_graphs // 8)
        self._gen_wheel(self.num_graphs // 8)
        self._gen_lollipop(self.num_graphs // 8)
        self._gen_hypercube(self.num_graphs // 8)
        self._gen_grid(self.num_graphs // 8)
        self._gen_clique(self.num_graphs // 8)
        self._gen_circular_ladder(self.num_graphs - len(self.graphs))
        # preprocess
        for i in range(self.num_graphs):
            # convert to Graph, and add self loops
            self.graphs[i] = add_self_loop(dgl_graph(self.graphs[i]))
        self.labels = F.tensor(np.array(self.labels).astype(np.int))

    def _gen_cycle(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.cycle_graph(num_v)
            self.graphs.append(g)
            self.labels.append(0)

    def _gen_star(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            # nx.star_graph(N) gives a star graph with N+1 nodes
            g = nx.star_graph(num_v - 1)
            self.graphs.append(g)
            self.labels.append(1)

    def _gen_wheel(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.wheel_graph(num_v)
            self.graphs.append(g)
            self.labels.append(2)

    def _gen_lollipop(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            path_len = np.random.randint(2, num_v // 2)
            g = nx.lollipop_graph(m=num_v - path_len, n=path_len)
            self.graphs.append(g)
            self.labels.append(3)

    def _gen_hypercube(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.hypercube_graph(int(math.log(num_v, 2)))
            g = nx.convert_node_labels_to_integers(g)
            self.graphs.append(g)
            self.labels.append(4)

    def _gen_grid(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            assert num_v >= 4, 'We require a grid graph to contain at least two ' \
                                   'rows and two columns, thus 4 nodes, got {:d} ' \
                                   'nodes'.format(num_v)
            n_rows = np.random.randint(2, num_v // 2)
            n_cols = num_v // n_rows
            g = nx.grid_graph([n_rows, n_cols])
            g = nx.convert_node_labels_to_integers(g)
            self.graphs.append(g)
            self.labels.append(5)

    def _gen_clique(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.complete_graph(num_v)
            self.graphs.append(g)
            self.labels.append(6)

    def _gen_circular_ladder(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.circular_ladder_graph(num_v // 2)
            self.graphs.append(g)
            self.labels.append(7)
