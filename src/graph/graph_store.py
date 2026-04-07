import networkx as nx


class GraphStore:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_triplets(self, triplets):
        for subj, rel, obj in triplets:
            self.graph.add_node(subj)
            self.graph.add_node(obj)
            self.graph.add_edge(subj, obj, relation=rel)

    def query(self, keyword):
        results = []

        for node in self.graph.nodes:
            if keyword.lower() in node.lower():
                neighbors = self.graph[node]

                for neighbor in neighbors:
                    relation = self.graph[node][neighbor]['relation']
                    results.append(f"{node} {relation} {neighbor}")

        return results