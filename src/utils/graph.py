import json
from pathlib import Path
from typing import Dict, List, Union
import networkx as nx

class ZoneGraph:
    """
    Representação do layout da loja num grafo.
    Pré-computa all-pairs shortest paths para lookups O(1).
    """
    __slots__ = ('_graph', '_shortest_paths')

    def __init__(self, zones_path: Union[str, Path] = "data/zones.json") -> None:
        self._graph = nx.Graph()
        
        with open(zones_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        zones_data = data.get("zones", {})
        
        for node, info in zones_data.items():
            self._graph.add_node(node)
            for adj_node, weight in info.get("walk_seconds", {}).items():
                self._graph.add_edge(node, adj_node, weight=weight)
                
        self._shortest_paths: Dict[str, Dict[str, int]] = dict(
            nx.all_pairs_dijkstra_path_length(self._graph, weight="weight")
        )

    def min_travel_time(self, zone_a: str, zone_b: str) -> int:
        """Lookup O(1) do tempo mínimo em segundos."""
        if zone_a == zone_b:
            return 0
        try:
            return self._shortest_paths[zone_a][zone_b]
        except KeyError:
            if zone_a not in self._shortest_paths or zone_b not in self._shortest_paths:
                raise ValueError(f"Zona desconhecida: {zone_a} ou {zone_b}")
            raise ValueError(f"Caminho impossível entre {zone_a} e {zone_b}")

    def are_adjacent(self, zone_a: str, zone_b: str) -> bool:
        return self._graph.has_edge(zone_a, zone_b)

    def get_all_zones(self) -> List[str]:
        return list(self._graph.nodes)
