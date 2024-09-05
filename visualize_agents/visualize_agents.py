from enum import Enum
from typing import Any, Dict, List

from streamlit_agraph import agraph, Node, Edge, Config
import streamlit as st


class NodeType(Enum):
    SEARCH = 0
    EVALUATION_FUNCTION = 1
    RL = 2
    HYBRID = 3
    ROOT = 4


AGENT_NODE_COLOR = '#1caa9c'
ROOT_COLOR = '#1caa9c'
EDGE_COLOR = '#1caa9c'


def _create_node(name: str, image_path: str, size=25, **kwargs) -> Node:
    return Node(id=name,
                title=name,
                label=name,
                size=size,
                shape='circularImage',
                image=image_path,
                **kwargs)


def _create_edge(parent_node: Node, child_node: Node, label: str = None, color=EDGE_COLOR, **kwargs) -> Edge:
    return Edge(source=parent_node.id,
                label=label,
                color=color,
                target=child_node.id,
                **kwargs)


def visualize_agents(nodes: List[Dict[str, Any]], images: Dict[NodeType, str], edges: List[Dict[str, Any]]) -> None:
    name_to_node = {}
    for node_dict in nodes:
        color = AGENT_NODE_COLOR if node_dict['is_agent'] else 'white'
        if node_dict['type'] == NodeType.ROOT:
            color = ROOT_COLOR
        name_to_node[node_dict['name']] = _create_node(node_dict['name'], images[node_dict['type']], color=color)

    created_edges = []
    for edge_dict in edges:
        child_node = name_to_node[edge_dict['child']]
        for parent in edge_dict['parents']:
            parent_node = name_to_node[parent]
            created_edges.append(_create_edge(parent_node, child_node))

    created_nodes = list(name_to_node.values())

    config = Config(width=2000,
                    height=1000,
                    directed=True,
                    physics=False,
                    hierarchical=True,
                    sortMethod='directed',
                    nodeSpacing=200,  # 100
                    treeSpacing=400,  # 200
    )

    agraph(nodes=created_nodes, edges=created_edges, config=config)


if __name__ == '__main__':
    images = {
        NodeType.SEARCH: 'https://i.ibb.co/4F5gYXx/graph.png',
        NodeType.RL: 'https://i.ibb.co/QJQ5PmZ/rl.png',
        NodeType.HYBRID: 'https://i.ibb.co/yhSy4Dh/hybrid.png',
        NodeType.EVALUATION_FUNCTION: 'https://i.ibb.co/47ZWP9v/evaluation-function.png',
        NodeType.ROOT: '',
    }
    nodes = [
        {'type': NodeType.ROOT, 'name': 'Agent Tree', 'is_agent': False},

        {'type': NodeType.EVALUATION_FUNCTION, 'name': 'AE Winning Possibilities', 'is_agent': False},
        {'type': NodeType.EVALUATION_FUNCTION, 'name': 'Probabilistic Estimator', 'is_agent': False},
        {'type': NodeType.EVALUATION_FUNCTION, 'name': 'Sub Boards Won', 'is_agent': False},
        # {'type': NodeType.EVALUATION_FUNCTION, 'name': 'Win-Loss Evaluation', 'is_agent': False},
        {'type': NodeType.EVALUATION_FUNCTION, 'name': 'Random Agent', 'is_agent': True},

        {'type': NodeType.SEARCH, 'name': 'MCTS', 'is_agent': True},
        {'type': NodeType.SEARCH, 'name': 'AlphaBeta', 'is_agent': False},

        {'type': NodeType.RL, 'name': 'DQN', 'is_agent': True},
        {'type': NodeType.RL, 'name': 'Reinforce', 'is_agent': True},

        {'type': NodeType.HYBRID, 'name': 'AB AE WP', 'is_agent': True},
        {'type': NodeType.HYBRID, 'name': 'AB PE', 'is_agent': True},
        {'type': NodeType.HYBRID, 'name': 'AB DQN', 'is_agent': True},
        {'type': NodeType.HYBRID, 'name': 'AB SBW', 'is_agent': True},
        {'type': NodeType.HYBRID, 'name': 'MCTS AE WP', 'is_agent': True},
        {'type': NodeType.HYBRID, 'name': 'MCTS PE', 'is_agent': True},
        {'type': NodeType.HYBRID, 'name': 'MCTS DQN', 'is_agent': True},
        {'type': NodeType.HYBRID, 'name': 'MCTS SBW', 'is_agent': True},
    ]

    edges = [
        {'parents': ['Agent Tree'], 'child': 'AE Winning Possibilities'},
        {'parents': ['Agent Tree'], 'child': 'Probabilistic Estimator'},
        {'parents': ['Agent Tree'], 'child': 'Sub Boards Won'},
        # {'parents': ['Agent Tree'], 'child': 'Win-Loss Evaluation'},
        {'parents': ['Agent Tree'], 'child': 'Random Agent'},
        {'parents': ['Agent Tree'], 'child': 'MCTS'},
        {'parents': ['Agent Tree'], 'child': 'AlphaBeta'},
        {'parents': ['Agent Tree'], 'child': 'DQN'},
        {'parents': ['Agent Tree'], 'child': 'Reinforce'},

        {'parents': ['AlphaBeta', 'AE Winning Possibilities'], 'child': 'AB AE WP'},
        {'parents': ['AlphaBeta', 'Probabilistic Estimator'], 'child': 'AB PE'},
        {'parents': ['AlphaBeta', 'DQN'], 'child': 'AB DQN'},
        {'parents': ['AlphaBeta', 'Sub Boards Won'], 'child': 'AB SBW'},
        {'parents': ['MCTS', 'AE Winning Possibilities'], 'child': 'MCTS AE WP'},
        {'parents': ['MCTS', 'Probabilistic Estimator'], 'child': 'MCTS PE'},
        {'parents': ['MCTS', 'DQN'], 'child': 'MCTS DQN'},
        {'parents': ['MCTS', 'Sub Boards Won'], 'child': 'MCTS SBW'},
    ]

    # TODO: assert edges are in nodes and vice versa

    visualize_agents(nodes, images, edges)

