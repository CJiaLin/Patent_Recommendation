from abc import ABCMeta, abstractmethod

class IGraph(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def nodes(self):
        pass

    @abstractmethod
    def edges(self):
        pass

    @abstractmethod
    def neighbors(self, node):
        pass

    @abstractmethod
    def has_node(self, node):
        pass

    @abstractmethod
    def add_node(self, node):
        pass

    @abstractmethod
    def has_edge(self, edge):
        pass

    @abstractmethod
    def add_edge(self, edge):
        pass

    @abstractmethod
    def add_edges(self, edges):
        pass

    @abstractmethod
    def edge_weight(self, edge):
        pass

    @abstractmethod
    def del_node(self, node):
        pass

class Graph(IGraph):

    DEFAULT_WEIGHT  = 0

    def __init__(self):

        self.node_neighbors = {}

    def __len__(self):

        return len(self.node_neighbors)

    def has_edge(self, edge):

        u, v = edge
        return (u in self.node_neighbors
                and v in self.node_neighbors
                and u in self.node_neighbors[v]
                and v in self.node_neighbors[u])

    def edge_weight(self, edge):

        u, v = edge
        return self.node_neighbors.get(u, {}).get(v, self.DEFAULT_WEIGHT)

    def neighbors(self, node):

        return list(self.node_neighbors[node])

    def has_node(self, node):

        return node in self.node_neighbors

    def add_edge(self, edge, wt=1):

        if wt == 0.0:
            if self.has_edge(edge):
                self.del_edge(edge)
            return
        u, v = edge
        if not self.has_node(u):
            self.add_node(u)
        if not self.has_node(v):
            self.add_node(v)
        if v not in self.node_neighbors[u]  and u not in self.node_neighbors[v]:
            self.node_neighbors[u][v] = wt
            if u != v:
                self.node_neighbors[v][u] = wt
        else:
            raise ValueError("Edge (%s, %s) already in graph" % (u, v))

    def add_edges(self, edges):
        for edge in edges:
            if len(edge) == 2:
                if not self.has_edge(edge):
                    self.add_edge(edge)
            if len(edge) == 3:
                if not self.has_edge(edge[:2]):
                    self.add_edge(edge[:2], edge[2])

    def add_node(self, node):

        if node in self.node_neighbors:
            raise ValueError("Node %s already in graph" % node)
        self.node_neighbors[node] = {}

    def nodes(self):

        return list(self.node_neighbors)

    def edges(self):

        return list(self.iter_edges())

    def iter_edges(self):

        for u in self.node_neighbors:
            for v in self.node_neighbors[u]:
                yield (u, v)

    def del_node(self, node):

        for each in self.neighbors(node):
            if each != node:
                self.del_edge((each, node))
        del self.node_neighbors[node]

    def del_edge(self, edge):

        u, v = edge
        del self.node_neighbors[u][v]
        if u != v:
            del self.node_neighbors[v][u]

def build_graph(sequence):

    graph = Graph()
    for item in sequence:
        if not graph.has_node(item):
            graph.add_node(item)
    return graph

def remove_unreachable_nodes(graph):

    for node in graph.nodes():
        if all(graph.edge_weight((node, other)) == 0 for other in graph.neighbors(node)):
            graph.del_node(node)

if __name__ == '__main__':
    gg = build_graph([])
    gg.add_edges([['Felidae', 'Lion'],['Lion', 'Tiger']])
    print(gg.edges())
    print(gg.nodes())