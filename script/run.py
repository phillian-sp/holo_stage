import os
import sys
import math
import pprint
import yaml
import pyrallis

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util
from nbfnet.models import EdgeGraphsNBFNetConfig, EdgeGraphsNBFNet
from nbfnet.util import DatasetConfig

from common_utils import MultiCounter, seed_everything, Logger, wrap_ruler
from dataclasses import dataclass, field
from typing import List


separator = ">" * 30
line = "-" * 30

METRIC = ["mr", "mrr", "hits@1", "hits@3", "hits@10", "hits@10_50"]


@dataclass
class MainConfig:
    seed: int = 1
    save_dir: str = "exp/sample/run1"
    use_wb: int = 0
    checkpoint: str = ""

    # Training parameters
    lr: float = 5.0e-3
    optimizer: str = "Adam"
    epochs: int = 30
    batch_size: int = 32
    eval_interval: int = 3

    # task specific
    num_negative: int = 64
    strict_negative: int = 1
    adversarial_temperature: float = 1.0
    metric: List[str] = field(default_factory=lambda: METRIC)

    # Model cfg
    nbf: EdgeGraphsNBFNetConfig = field(default_factory=EdgeGraphsNBFNetConfig)

    # Dataset cfg
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Wandb
    wb_exp: str = ""
    wb_run: str = ""
    wb_group: str = ""

    @property
    def cfg_path(self):
        return os.path.join(self.save_dir, "cfg.yaml")

    @property
    def log_path(self):
        return os.path.join(self.save_dir, "train.log")

    @property
    def model_path(self):
        return os.path.join(self.save_dir, "model.pt")

    def __post_init__(self):
        seed_everything(self.seed)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if self.save_dir[-1] == "/":
            self.save_dir = self.save_dir[:-1]
        if "seed" not in self.save_dir:
            self.save_dir += f"_seed{self.seed}"
        if self.wb_exp == "" and self.use_wb:
            self.wb_exp = self.save_dir.split("/")[-2]
        if self.wb_run == "" and self.use_wb:
            self.wb_run = self.save_dir.split("/")[-1]
        if self.wb_group == "" and self.use_wb:
            self.wb_group = "_".join(
                [w for w in self.wb_run.split("_") if "seed" not in w]
            )


class Workspace:
    def __init__(self, cfg: MainConfig):
        self.cfg = cfg
        sys.stdout = Logger(cfg.log_path, print_to_stdout=True)
        pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
        print(wrap_ruler("config"))
        with open(cfg.cfg_path, "r") as f:
            print(f.read(), end="")
        print(wrap_ruler(""))
        self.cfg_dict = yaml.safe_load(open(cfg.cfg_path, "r"))
        self.stat = MultiCounter(
            self.cfg.save_dir,
            bool(self.cfg.use_wb),
            wb_exp_name=self.cfg.wb_exp,
            wb_run_name=self.cfg.wb_run,
            wb_group_name=self.cfg.wb_group,
            config=self.cfg_dict,
        )

        # print infos
        print(wrap_ruler("info"))
        print(f"Working dir: {self.cfg.save_dir}")
        print(f"Random seed: {self.cfg.seed}")
        print(f"Train categories: {self.cfg.dataset.train_categories}")
        print(f"Test categories: {self.cfg.dataset.test_categories}")
        print(f"Feature method: {self.cfg.model.feature_method}")
        print(f"Config file: {self.cfg.cfg_path}")
        print(wrap_ruler(""))

        # build dataset
        self.dataset_list, self.num_relations = util.build_dataset(self.cfg)
        self.model: EdgeGraphsNBFNet = util.build_model(self.cfg)
        self.model = self.model.to(self.cfg.device)
        self.train_data_list, self.valid_data_list, self.test_data_list = (
            self.dataset_list
        )
        self.train_data_dict = {
            self.cfg.dataset.train_categories[i]: self.train_data_list[i].to(
                self.cfg.device
            )
            for i in range(len(self.train_data_list))
        }
        self.valid_data_dict = {
            self.cfg.dataset.train_categories[i]: self.valid_data_list[i].to(
                self.cfg.device
            )
            for i in range(len(self.valid_data_list))
        }
        self.test_data_dict = {
            self.cfg.dataset.test_categories[i]: self.test_data_list[i].to(
                self.cfg.device
            )
            for i in range(len(self.test_data_list))
        }

    def train_and_validate(self):
        if self.cfg.epochs == 0:
            return

        train_loaders = {}
        for name, train_data in self.train_data_dict.items():
            train_triplets = torch.cat(
                [train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]
            ).t()
            sampler = torch_data.DistributedSampler(train_triplets)
            train_loaders[name] = torch_data.DataLoader(
                train_triplets, self.cfg.batch_size, sampler=sampler
            )

        optimizer: optim.Optimizer = getattr(optim, self.cfg.optimizer)(
            self.model.parameters(), lr=self.cfg.lr
        )

        best_result = float("-inf")
        best_epoch = -1

        for epoch in range(0, self.cfg.epochs):  # for each eval_interval
            self.model.train()
            print(wrap_ruler(f"Epoch {epoch} begin"))

            losses = []
            sampler.set_epoch(epoch)
            for dataset_name, train_loader in train_loaders.items():
                print(f"Start training on {dataset_name}")
                for batch in train_loader:  # for each batch in a given category
                    batch_size = batch.size(0)
                    if hasattr(self.cfg.nbf, "edge_embed_dim"):
                        edge_embed = self.cfg.nbf.edge_embed_dim
                    else:
                        edge_embed = None
                    # batch: [batch_size, 3] -> [batch_size, num_negative+1, 3]
                    batch = tasks.negative_sampling(
                        self.train_data_dict[dataset_name],
                        batch,
                        self.cfg.num_negative,
                        edge_embed_dim=edge_embed,
                        strict=self.cfg.strict_negative,
                    )
                    # pred: [batch_size, num_negative+1]
                    pred = self.model(self.train_data_dict[dataset_name], batch)
                    # target: [batch_size, num_negative+1]
                    target = torch.zeros_like(pred)
                    target[:, 0] = 1

                    loss = F.binary_cross_entropy_with_logits(
                        pred, target, reduction="none"
                    )

                    neg_weight = torch.ones_like(pred)
                    if self.cfg.adversarial_temperature > 0:
                        with torch.no_grad():
                            neg_weight[:, 1:] = F.softmax(
                                pred[:, 1:] / self.cfg.adversarial_temperature,
                                dim=-1,
                            )
                    else:
                        neg_weight[:, 1:] = 1 / self.cfg.num_negative

                    loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                    loss = loss.mean()

                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    losses.append(loss.item())
                    self.stat[f"loss/{dataset_name}"].append(
                        loss.item(), count=batch_size
                    )

                # end of training on a category
                avg_loss = sum(losses) / len(losses)
                print(f"average binary cross entropy: {avg_loss}")
                print(wrap_ruler(f"Epoch {epoch} end"))

            if (epoch + 1) % self.cfg.eval_interval == 0:
                # end of training on all categories in a eval_interval
                print("Save checkpoint to model_epoch_%d.pth" % epoch)
                state = {
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(state, "model_epoch_%d.pth" % epoch)

                result = self.test_list(mode="valid")
                if result > best_result:
                    best_result = result
                    best_epoch = epoch

                test_result = self.test_list(mode="test")

            # end of training on all categories in a epoch
            self.stat[f"other/epoch"].append(epoch)
            self.stat.summary(epoch, reset=True)

        print("Load checkpoint from model_epoch_%d.pth" % best_epoch)
        state = torch.load(
            "model_epoch_%d.pth" % best_epoch
        )  # , map_location=self.device)
        self.model.load_state_dict(state["model"])

        result = self.test_list(mode="valid")
        test_result = self.test_list(mode="test")

    def test_list(self, mode="test"):
        print(wrap_ruler(f"Test on {mode}"))
        if mode == "valid":
            data_dict = self.valid_data_dict
        elif mode == "test":
            data_dict = self.test_data_dict
        else:
            raise ValueError(f"Invalid mode: {mode}")

        mrr_list = []
        for name, test_data in data_dict.items():
            print(f"Start testing on {name}")
            mrr = self.test(name, test_data)
            mrr_list.append(mrr)

        ave_mrr = sum(mrr_list) / len(mrr_list)
        print(f"Average MRR: {ave_mrr}")
        print(wrap_ruler(""))
        return ave_mrr

    @torch.no_grad()
    def test(self, dataset_name, test_data):
        # TODO: see if we need to use filtered_data
        filtered_data = None

        test_triplets = torch.cat(
            [test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]
        ).t()
        sampler = torch_data.DistributedSampler(test_triplets)
        test_loader = torch_data.DataLoader(
            test_triplets, self.cfg.batch_size, sampler=sampler
        )

        self.model.eval()
        rankings = []
        num_negatives = []
        for batch in test_loader:
            t_batch, h_batch = tasks.all_negative(test_data, batch)
            if hasattr(self.cfg.nbf, "edge_embed_dim"):
                edge_embed = self.cfg.nbf.edge_embed_dim
            else:
                edge_embed = None
            if filtered_data is None:
                t_mask, h_mask = tasks.strict_negative_mask(
                    test_data, batch, edge_embed
                )
            else:
                t_mask, h_mask = tasks.strict_negative_mask(
                    filtered_data, batch, edge_embed
                )
            t_pred = self.model(test_data, t_batch)
            h_pred = self.model(test_data, h_batch)
            pos_h_index, pos_t_index, pos_r_index = batch.t()
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
            num_t_negative = t_mask.sum(dim=-1)
            num_h_negative = h_mask.sum(dim=-1)

            rankings += [t_ranking, h_ranking]
            num_negatives += [num_t_negative, num_h_negative]

        all_ranking = torch.cat(rankings)
        all_num_negative = torch.cat(num_negatives)

        for metric in self.cfg.metric:
            if metric == "mr":
                score = all_ranking.float().mean()
            elif metric == "mrr":
                score = (1 / all_ranking.float()).mean()
            elif metric.startswith("hits@"):
                values = metric[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (all_ranking - 1).float() / all_num_negative
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = (
                            math.factorial(num_sample - 1)
                            / math.factorial(i)
                            / math.factorial(num_sample - i - 1)
                        )
                        score += (
                            num_comb
                            * (fp_rate**i)
                            * ((1 - fp_rate) ** (num_sample - i - 1))
                        )
                    score = score.mean()
                else:
                    score = (all_ranking <= threshold).float().mean()
            print("%s: %g" % (metric, score))
            self.stat[f"{dataset_name}/{metric}"].append(score)
        mrr = (1 / all_ranking.float()).mean()

        return mrr


if __name__ == "__main__":
    cfg = MainConfig()
    workspace = Workspace(cfg)
    workspace.train_and_validate()
