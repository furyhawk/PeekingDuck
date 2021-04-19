""" Reads and manipulations configuration files
"""

import os
import yaml
from typing import Any, Dict, List
from peekingduck.pipeline.nodes.node import AbstractNode

class ConfigLoader:
    """ Reads configuration and returns configuration required to
    create the nodes for the project
    """
    def __init__(self, nodes: List[AbstractNode], node_yml: str = None):
        self._master_node_config = {}
        self._rootdir = os.path.join(
            os.path.dirname(os.path.abspath(__file__))
        )
        if node_yml:
            node_config = self._load_from_path(node_yml)
            self._master_node_config = node_config
        else:
            self._load_from_node_list(nodes)

    def _load_from_node_list(self, nodes: List[AbstractNode]) -> None:
        """load node_configs from a list of nodes, configs is expected to be at level of peekingduck"""
        for node in nodes:
            config_filename = node.replace('.','_') + '.yml'
            filepath = os.path.join(self._rootdir, 'configs',config_filename)
            node_config = self._load_from_path(filepath)
            self._master_node_config[node] = node_config


    def _load_from_path(self, filepath: str) -> None:
        """load node_configs directly from a custom node_config"""
        with open(filepath) as file:
            node_config = yaml.load(file, Loader=yaml.FullLoader)
        return node_config

    def get(self, item: str) -> Dict[str, Any]:
        """Get item from configuration read from the filepath,
        item refers to the node item configuration you are trying to get"""

        node = self._master_node_config[item]

        # some models require the knowledge of where the root is for loading
        node['root'] = self._rootdir
        return node
