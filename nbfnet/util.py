import os
import sys
import ast
import time
import logging
import argparse
import shlex
import subprocess
from pathlib import Path

import yaml
import jinja2
from jinja2 import meta
import easydict
from dataclasses import dataclass, field
from typing import List

import torch
from torch import distributed as dist
from torch_geometric.data import Data
from torch_geometric.datasets import RelLinkPredDataset, WordNet18RR
from torch_geometric.data import InMemoryDataset

from torch_geometric.utils import to_networkx
import networkx as nx


from nbfnet import models, datasets
from nbfnet import load_ecommerce


def analyze_connected_components(data):
    """
    Analyzes the number of connected components in the graph formed by edge_index
    and their sizes (number of nodes and edges in the connected components).

    Parameters:
    data (Data): A PyTorch Geometric Data object with edge_index.

    Returns:
    tuple: (num_connected_components, components_info)
           - num_connected_components (int): Number of connected components.
           - components_info (list): List of tuples containing number of nodes and edges
                                     in each connected component.
    """
    # Convert the PyTorch Geometric data to a NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Find connected components
    connected_components = list(nx.connected_components(G))

    # Analyze connected components
    components_info = []
    for component in connected_components:
        subgraph = G.subgraph(component)
        num_nodes = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()
        components_info.append((num_nodes, num_edges))

    num_connected_components = len(connected_components)
    print(f"Number of connected components: {num_connected_components}")
    print(f"Sizes of connected components (nodes, edges): {components_info}")


def count_unique_target_nodes_not_in_edge_index(data):
    """
    Counts how many node IDs in target_edge_index do not appear in edge_index at all.

    Parameters:
    data (Data): A PyTorch Geometric Data object with edge_index and target_edge_index.

    Returns:
    int: Number of node IDs in target_edge_index not in edge_index.
    """
    # Extract unique node IDs from edge_index and target_edge_index
    unique_nodes_in_edge_index = set(data.edge_index.reshape(-1).tolist())
    unique_nodes_in_target_edge_index = set(data.target_edge_index.reshape(-1).tolist())

    # Find nodes in target_edge_index that do not appear in edge_index
    nodes_not_in_edge_index = unique_nodes_in_target_edge_index - unique_nodes_in_edge_index
    num_nodes_not_in_edge_index = len(nodes_not_in_edge_index)

    print(f"Number of node IDs in target_edge_index not in edge_index: {num_nodes_not_in_edge_index}")


@dataclass
class DatasetConfig:
    name: str = "Indecommerce"
    csv_file_path: str = "/home/phillipmiao/feature_invariant/NBFNet-PyG/data/"
    root: str = "/home/phillipmiao/feature_invariant/NBFNet-PyG/data/row1000_train4/"
    num_rows: int = 1000
    train_categories: List[str] = field(
        default_factory=lambda: [
            "appliances.kitchen.refrigerators",
            "furniture.bedroom.bed",
            "electronics.smartphone",
            "apparel.shoes",
        ]
    )
    test_categories: List[str] = field(default_factory=lambda: ["computers.desktop"])


from script.run import MainConfig


def build_dataset(cfg: MainConfig):
    cls = cfg.dataset.name
    if cls == "Indecommerce":
        train_data_list = []
        valid_data_list = []
        for i in range(len(cfg.dataset.train_categories)):
            train_category = cfg.dataset.train_categories[i]
            dataset_name = "ecommerce"
            if train_category == "hm":
                train_category = None
                dataset_name = "hm"
            print("dataset_name: ", dataset_name)
            print("train_category: ", train_category)
            dataset = load_ecommerce.MyGraphDataset(
                cfg.dataset.csv_file_path,
                root=cfg.dataset.root,
                num_rows=cfg.dataset.num_rows,
                train_category=train_category,
                feature_method="ours",
                dataset_name=dataset_name,
                p_value=cfg.edgegraph.use_p_value,
                input_dim=cfg.edgegraph.nbf.input_dim,
            )
            print(dataset[0])
            # analyze_connected_components(dataset[0])
            # count_unique_target_nodes_not_in_edge_index(dataset[0])
            print(dataset[1])
            # analyze_connected_components(dataset[1])
            # count_unique_target_nodes_not_in_edge_index(dataset[1])
            train_data_list.append(dataset[0])
            valid_data_list.append(dataset[1])
        test_data_list = []
        for i in range(len(cfg.dataset.test_categories)):
            test_category = cfg.dataset.test_categories[i]
            dataset_name = "ecommerce"
            if test_category == "hm":
                test_category = None
                dataset_name = "hm"

            dataset = load_ecommerce.MyGraphDataset(
                cfg.dataset.csv_file_path,
                root=cfg.dataset.root,
                num_rows=cfg.dataset.num_rows,
                test_category=test_category,
                feature_method="ours",
                mode="test",
                dataset_name=dataset_name,
                p_value=cfg.edgegraph.use_p_value,
                input_dim=cfg.edgegraph.nbf.input_dim,
            )

            print(dataset[0])
            # analyze_connected_components(dataset[0])
            # count_unique_target_nodes_not_in_edge_index(dataset[0])
            test_data_list.append(dataset[0])

        dataset_list = (train_data_list, valid_data_list, test_data_list)
        # num_relations = train_data_list[0].edge_type.max().item() + 1
        # define num_relations by looping through all the data (train and test) list and finding the max edge_type
        num_relations = max(
            [data.edge_type.max().item() + 1 for data in train_data_list + valid_data_list + test_data_list]
        )
    else:
        raise ValueError("Unknown dataset `%s`" % cls)

    print("%s dataset" % cls)
    train_data_list, valid_data_list, test_data_list = dataset_list
    print(
        "#train: %d, #valid: %d, #test: %d"
        % (
            sum([data.target_edge_index.shape[1] for data in train_data_list]),
            sum([data.target_edge_index.shape[1] for data in valid_data_list]),
            sum([data.target_edge_index.shape[1] for data in test_data_list]),
        )
    )

    return dataset_list, num_relations


def build_model(num_relations, cfg: MainConfig) -> models.EdgeGraphsNBFNet:
    model = models.EdgeGraphsNBFNet(num_relations, cfg.edgegraph)
    if cfg.checkpoint != "":
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
    return model


def git_logs(run_directory: str):
    try:
        script_directory = Path(__file__).resolve().parent.parent.parent
        dirty = subprocess.call(shlex.split("git diff-index --quiet HEAD --"))
        if dirty != 0:
            with open(Path(run_directory) / "dirty.diff", "w") as f:
                subprocess.call(shlex.split("git diff"), stdout=f, stderr=f)
        git_hash = subprocess.check_output(shlex.split("git describe --always"), cwd=script_directory).strip().decode()
        print(f"Git hash: {git_hash}")
    except subprocess.CalledProcessError:
        print("Could not retrieve git hash")
