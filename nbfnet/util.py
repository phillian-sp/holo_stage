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


# logger = logging.getLogger(__file__)


# def detect_variables(cfg_file):
#     with open(cfg_file, "r") as fin:
#         raw = fin.read()
#     env = jinja2.Environment()
#     tree = env.parse(raw)
#     vars = meta.find_undeclared_variables(tree)
#     return vars


# def load_config(cfg_file, context=None):
#     with open(cfg_file, "r") as fin:
#         raw = fin.read()
#     template = jinja2.Template(raw)
#     instance = template.render(context)
#     cfg = yaml.safe_load(instance)
#     cfg = easydict.EasyDict(cfg)
#     return cfg


# def literal_eval(string):
#     try:
#         return ast.literal_eval(string)
#     except (ValueError, SyntaxError):
#         return string


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
#     parser.add_argument(
#         "-s", "--seed", help="random seed for PyTorch", type=int, default=1024
#     )
#     parser.add_argument(
#         "--checkpoint", help="checkpoint file to resume training", default=None
#     )

#     args, unparsed = parser.parse_known_args()
#     # get dynamic arguments defined in the config file
#     vars = detect_variables(args.config)
#     parser = argparse.ArgumentParser()
#     for var in vars:
#         parser.add_argument("--%s" % var, required=True)
#     vars = parser.parse_known_args(unparsed)[0]
#     vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}

#     return args, vars


# # def get_root_logger(file=True):
# #     format = "%(asctime)-10s %(message)s"
# #     datefmt = "%H:%M:%S"
# #     logging.basicConfig(format=format, datefmt=datefmt)
# #     logger = logging.getLogger("")
# #     logger.setLevel(logging.INFO)

# #     if file:
# #         handler = logging.FileHandler("log.txt")
# #         format = logging.Formatter(format, datefmt)
# #         handler.setFormatter(format)
# #         logger.addHandler(handler)

# #     return logger


# def get_rank():
#     if dist.is_initialized():
#         return dist.get_rank()
#     if "RANK" in os.environ:
#         return int(os.environ["RANK"])
#     return 0


# def get_world_size():
#     if dist.is_initialized():
#         return dist.get_world_size()
#     if "WORLD_SIZE" in os.environ:
#         return int(os.environ["WORLD_SIZE"])
#     return 1


# def synchronize():
#     if get_world_size() > 1:
#         dist.barrier()


# def get_device(cfg):
#     if cfg.train.gpus:
#         device = torch.device(cfg.train.gpus[get_rank()])
#         print(device)
#     else:
#         device = torch.device("cpu")
#     return device


# def create_working_directory(cfg):
#     file_name = "working_dir.tmp"
#     world_size = get_world_size()
#     if cfg.train.gpus is not None and len(cfg.train.gpus) != world_size:
#         error_msg = "World size is %d but found %d GPUs in the argument"
#         if world_size == 1:
#             error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
#         raise ValueError(error_msg % (world_size, len(cfg.train.gpus)))
#     if world_size > 1 and not dist.is_initialized():
#         dist.init_process_group("nccl", init_method="env://")

#     working_dir = os.path.join(
#         os.path.expanduser(cfg.output_dir),
#         cfg.model["class"] + f"-{cfg.model['feature_method']}",
#         cfg.dataset["class"],
#         time.strftime("%m-%d"),
#         str(time.time_ns()),  # timestamp acting as an experiment ID
#     )

#     # synchronize working directory
#     if get_rank() == 0:
#         with open(file_name, "w") as fout:
#             fout.write(working_dir)
#         os.makedirs(working_dir)
#     synchronize()
#     if get_rank() != 0:
#         with open(file_name, "r") as fin:
#             working_dir = fin.read()
#     synchronize()
#     if get_rank() == 0:
#         os.remove(file_name)

#     os.chdir(working_dir)
#     return working_dir


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
    nodes_not_in_edge_index = (
        unique_nodes_in_target_edge_index - unique_nodes_in_edge_index
    )
    num_nodes_not_in_edge_index = len(nodes_not_in_edge_index)

    print(
        f"Number of node IDs in target_edge_index not in edge_index: {num_nodes_not_in_edge_index}"
    )


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
            [
                data.edge_type.max().item() + 1
                for data in train_data_list + valid_data_list + test_data_list
            ]
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
        git_hash = (
            subprocess.check_output(
                shlex.split("git describe --always"), cwd=script_directory
            )
            .strip()
            .decode()
        )
        print(f"Git hash: {git_hash}")
    except subprocess.CalledProcessError:
        print("Could not retrieve git hash")
