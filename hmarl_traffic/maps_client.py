import json
import networkx as nx
from hmarl_traffic import config

class MapsClient:
    """
    Mengelola peta jaringan internal dan
    menjalankan algoritma pathfinding untuk V2I.
    """
    def __init__(self, graph_map_path=config.GRAPH_MAP_PATH):
        self.graph = self._load_graph(graph_map_path)
        print("MapsClient siap. Graf NetworkX berhasil dibuat.")

    def _load_graph(self, file_path):
        """Memuat file JSON graf dan mengubahnya menjadi objek NetworkX."""
        try:
            with open(file_path, 'r') as f:
                graph_data = json.load(f)
            
            G = nx.DiGraph() # Graf berarah
            G.add_nodes_from(graph_data["nodes"])
            # Asumsi semua edge memiliki bobot 1
            G.add_edges_from([(u, v, {"weight": 1}) for u, v in graph_data["edges"]])
            return G
        except Exception as e:
            print(f"Error memuat {file_path}: {e}")
            return nx.DiGraph()

    def get_route(self, start_agent_id, end_agent_id):
        """
        Menjalankan algoritma pathfinding pada Peta Internal.
        """
        if start_agent_id not in self.graph or end_agent_id not in self.graph:
            print(f"Error: Node {start_agent_id} or {end_agent_id} tidak ada di graf.")
            return []
            
        try:
            # Menggunakan Dijkstra untuk menemukan path terpendek
            path = nx.dijkstra_path(self.graph, start_agent_id, end_agent_id, weight="weight")
            return path
        except nx.NetworkXNoPath:
            print(f"Tidak ada rute ditemukan dari {start_agent_id} ke {end_agent_id}")
            return []