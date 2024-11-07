import os
import argparse
from typing import Dict

import torch
from torch_geometric.data import Batch, Data, InMemoryDataset


from .data_utils import (
    get_feature_map,
    build_edge_graph,
    build_product_product_graph,
    compute_probabilities,
    get_user_product_graph,
    load_max_connected,
)

from .feature_method import encode_input_features



class MyGraphDataset(InMemoryDataset):
    def __init__(
        self,
        csv_file_path,
        root,
        num_rows,
        dataset_name,
        feature_method="ours",
        p_value=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        train_category=None,
        test_category=None,
        mode="train",
        input_dim=256,
    ):
        self.csv_file_path = csv_file_path
        self.num_rows = num_rows
        self.train_category = train_category
        self.test_category = test_category
        self.dataset_name = dataset_name
        self.mode = mode
        self.feature_method = feature_method
        self.p_value = p_value
        self.input_dim = input_dim
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        if self.train_category is not None:
            root = root + '/' + train_category
            self.train_feature_csv_file_path = os.path.join(base_dir,"data", f"featured_{train_category}.csv") 
        elif self.test_category is not None:
            root = root + '/' + test_category
            self.test_feature_csv_file_path = os.path.join(base_dir,"data", f"featured_{test_category}.csv")
        root = root + '_' + mode + '_' + feature_method + '_' + str(num_rows)
        if p_value:
            root = root + '_pvalue'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return [self.csv_file_path]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        # load data - data.x is a List[Dict[str, torch.Tensor]] of raw features
        data_dict, feature_keys, feat_metadata = self.load_data_from_csv()

        # applies the feature encoding method of choice to data.x
        data_dict = encode_input_features(data_dict, self.feature_method, feat_metadata, self.input_dim)
        
        if self.pre_transform is not None:
            for key, data in data_dict.items():
                data_dict[key] = self.pre_transform(data)
        
        torch.save(
            (self.collate(list(data_dict.values()))), self.processed_paths[0]
        )


    def load_data_from_csv(self, ) -> Dict[str, Data]:
        """
        Loads dict of Data objects from the processed file.
        Does **not** apply FeatureEncoder method.
        """
        feature_csv_path_dict = {}
        if self.dataset_name == "ecommerce":
            up_data, _, up_product2node, feature_keys, x_map = self.get_up_data(
                                                            category=self.test_category if self.mode == "test" else self.train_category,
                                                            feature_csv_path=self.test_feature_csv_file_path if self.mode == "test" else self.train_feature_csv_file_path,
                                                            # start_row=self.num_rows if self.mode == "test" else 0) # test on hm
                                                            start_row=0) # test within ecommerce (for price experiment)
        elif self.dataset_name == "hm":
            # feature_csv_path_dict = {"customer": f"{self.csv_file_path}hm/part_customers.csv", 
            #                         "article": f"{self.csv_file_path}hm/part_articles.csv"}
            #use the following dict to only get the top k frequent customers and articles
            feature_csv_path_dict = {"customer": f"{self.csv_file_path}hm/top_k_frequent_customers.csv",
                                    "article": f"{self.csv_file_path}hm/top_k_frequent_articles.csv"}
            up_data, _, up_product2node, feature_keys, x_map = self.get_up_data_hm(
                                                            category=self.test_category if self.mode == "test" else self.train_category,
                                                            feature_csv_path_dict=feature_csv_path_dict,
                                                            start_row=self.num_rows if self.mode == "test" else 0)
        ## Train / Val split
        num_edges = up_data.edge_index.size(1)
        observable_end = int(0.8 * num_edges)

        # TODO(joshrob): fix random seed?
        # TODO(okyangyishen): permute the edges after selecting the observable edges
        perm = torch.randperm(num_edges)

        # train edges
        edge_index = up_data.edge_index[:, perm[:observable_end].sort().values]
        target_edge_index = edge_index.clone()
        edge_attr = up_data.edge_attr[perm[:observable_end].sort().values]
        target_edge_type = edge_attr.clone()

        # val / test edges
        unobservable_edge_index = up_data.edge_index[:, perm[observable_end:].sort().values]
        unobservable_edge_attr = up_data.edge_attr[perm[observable_end:].sort().values]

        # add edges to user-product graph
        up_data.edge_index = edge_index
        up_data.edge_attr = edge_attr

        ## Get product-product graph
        if len(feature_csv_path_dict) < 3: # building product-product graph for both ecommerce and hm
            pp_data = build_product_product_graph(up_data, up_product2node)
        
        ## Add reverse edges (product to user)
        up_data.edge_index = torch.cat(
            [up_data.edge_index, up_data.edge_index.flip(0)], dim=-1
        )
        up_data.edge_attr = torch.cat(
            [up_data.edge_attr, up_data.edge_attr + up_data.edge_attr.max() + 1]
        )

        ## Create product product edge types
        if len(feature_csv_path_dict) < 3:
            pp_edge_attr = torch.full_like(
                pp_data.edge_index[0], fill_value=up_data.edge_attr.max() + 1
            )
            print("edge_index shape before concat", up_data.edge_index.shape)
            edge_index = torch.cat([up_data.edge_index, pp_data.edge_index], dim=-1)
            print("edge_index shape after concat", edge_index.shape)
            edge_type = torch.cat([up_data.edge_attr, pp_edge_attr])
            original_edge_type = torch.cat([up_data.edge_attr, pp_edge_attr])
        else:
            edge_index = up_data.edge_index
            edge_type = up_data.edge_attr
            original_edge_type = up_data.edge_attr

        # Construct base data
        # train / val / test data have different target edges
        base_data = Data(
            x=None, # added later by FeatureEncoder
            edge_index=edge_index,
            edge_type=edge_type,
            original_edge_type=original_edge_type,
            num_nodes=up_data.num_nodes,
        )


        #if self.mode == "train":
        # Note that train_data.edge_index[:, -(batch.batch.max() + 1): ] corresponds to pp_data.edge_index
        # Note that batch is now defined later, but I kept Beatrice's comment above just in case she neeeds that
        train_data = base_data.clone()
        train_data.target_edge_index = target_edge_index
        train_data.target_edge_type = target_edge_type

        valid_data = base_data.clone()
        valid_data.target_edge_index = unobservable_edge_index
        valid_data.target_edge_type = unobservable_edge_attr

        data_dict = {"train": train_data, "valid": valid_data}
        
        # TODO(joshrob): remove valid naming throughout
        if self.mode == "test":
            data_dict["test"] = valid_data 
            del data_dict["train"]
            del data_dict["valid"]



        # Stored any extra data needed for feature encoding
        feat_method_metadata = {}
        if self.feature_method == "ours":
            # add pp_data as data to encode
            if len(feature_csv_path_dict) < 3:
                feat_method_metadata["data_for_computing_edge_graphs"] = pp_data
            else:
                feat_method_metadata["data_for_computing_edge_graphs"] = up_data
            feat_method_metadata["feature_keys"] = feature_keys    
            feat_method_metadata["p_value"] = self.p_value

        elif self.feature_method == "normalized":
            feat_method_metadata["up_data"] = up_data
            feat_method_metadata["feature_keys"] = feature_keys

        elif self.feature_method == "price":
            feat_method_metadata["up_data"] = up_data
        
        elif self.feature_method == "llm":
            feat_method_metadata["up_data"] = up_data
            feat_method_metadata["feature_keys"] = feature_keys
            feat_method_metadata["x_map"] = x_map   

        elif self.feature_method == "raw":
            feat_method_metadata["up_data"] = up_data

        return data_dict, feature_keys, feat_method_metadata



    # get connected user product data with unprocessed features
    # assumes ecomerce
    #TODO(joshrob): add more documentation
    def get_up_data(self, category: str, 
                    feature_csv_path: str = None,
                    start_row: int = 0) -> Data:

        if feature_csv_path is not None:
            unwanted_columns = ["product_id", "category_code", "event_time"] # can remove more columns, but keep id in the first place
            feature_map, feature_keys = get_feature_map(feature_csv_path, unwanted_columns=unwanted_columns)

        # event_type_mapping = {
        #     "view": 0,
        #     "cart": 1,
        #     "purchase": 2,
        #     "remove_from_cart": 3,
        # }  # Mapping event types to integers

        # only filter out purchase events
        # event_type_mapping = {
        #     "purchase": 0,
        # }

        # mark all events as purchase
        # event_type_mapping = {
        #     "view": 0,
        #     "cart": 0,
        #     "purchase": 0,
        #     "remove_from_cart": 0,
        # }

        # mark purchase as first so hm will have the same purchase relation mapping
        event_type_mapping = {
            "view": 1,
            "cart": 2,
            "purchase": 0,
            "remove_from_cart": 3,
        }

        # Construct user product data
        up_data, up_user2node, up_product2node, feature_keys, x_map = get_user_product_graph(
            f"{self.csv_file_path}2019-Nov.csv", 
            # use the filtered csv file according to category, not used right now
            # f"{self.csv_file_path}{category.replace('.', '_')}_frequent.csv",
            start_row=start_row, 
            end_row=start_row + self.num_rows, 
            category=category, 
            feature_maps=[feature_map], 
            feature_keys=[feature_keys],
            category_column="category_code", 
            product_id_column="product_id", 
            user_id_column="user_id",
            event_type_column="event_type", 
            event_type_mapping=event_type_mapping
        )
        up_data, up_user2node, up_product2node = load_max_connected(
            up_data, up_user2node, up_product2node
        )

        return up_data, up_user2node, up_product2node, feature_keys, x_map
    
    def get_up_data_hm(self, category: str, 
                    feature_csv_path_dict: Dict[str, str] = None,
                    start_row: int = 0) -> Data:
        
        if feature_csv_path_dict is not None:
            customer_feature_keys = {}
            article_feature_keys = {}
            customer_feature_map = {}
            article_feature_map = {}
            for key in feature_csv_path_dict:
                if key == "customer":
                    # don't want customer_id,FN,Active,postal_code
                    unwanted_columns = ["customer_id", "FN", "Active", "postal_code"]
                    customer_feature_map, customer_feature_keys = get_feature_map(feature_csv_path_dict[key], unwanted_columns=unwanted_columns)
                elif key == "article":
                    # don't want product_code,prod_name,product_type_no,graphical_appearance_no,colour_group_code,perceived_colour_value_name,perceived_colour_master_id,department_no,index_code,index_group_no,section_no,garment_group_no,detail_desc
                    unwanted_columns = ["article_id", "product_code", "prod_name", "product_type_no", "graphical_appearance_no", "colour_group_code", "perceived_colour_value_id", "perceived_colour_master_id", "department_no", "index_code", "index_group_no", "section_no", "garment_group_no", "detail_desc"]
                    article_feature_map, article_feature_keys = get_feature_map(feature_csv_path_dict[key], unwanted_columns=unwanted_columns)
                else:
                    raise ValueError("Invalid key in feature_csv_path_dict")
            feature_maps = [customer_feature_map, article_feature_map]
            feature_keys = [customer_feature_keys, article_feature_keys]
            
            up_data, up_user2node, up_product2node, feature_keys, x_map = get_user_product_graph(
                f"{self.csv_file_path}hm/part_transactions.csv",    # file too large to upload to github, ask Yangyi for local share
                start_row=start_row, 
                end_row=start_row + self.num_rows, 
                category=category, 
                feature_maps=feature_maps,
                feature_keys=feature_keys,
                category_column="product_type_name", 
                product_id_column="article_id", 
                user_id_column="customer_id",
            )
            up_data, up_user2node, up_product2node = load_max_connected(
                up_data, up_user2node, up_product2node
            )

            return up_data, up_user2node, up_product2node, feature_keys, x_map


    
# if __name__ == "__main__":
    # # # Your CSV file path
    # csv_file_path = "data/2019-Nov.csv"

    # # add a parser to parse an argument of the number of rows
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--num_rows", type=int, default=100)
    # parser.add_argument("--root", type=str, default="hmdata/")
    # parser.add_argument("--train_category", type=str, default=None)
    # parser.add_argument("--test_category", type=str, default=None)
    # parser.add_argument("--mode", type=str, default="train")
    # args = parser.parse_args()

    # dataset = MyGraphDataset(csv_file_path, root=args.root, num_rows=args.num_rows, 
    #                         train_category=[args.train_category],
    #                         test_category=[args.test_category],
    #                         dataset_name="ecommerce", mode=args.mode)
    # data = dataset[0]
    # print(data)
    # print(data.num_edges)
    # data = dataset[1]
    # print(data)

    # test HM data loading
    # csv_file_path = "data/hm/part_transactions.csv"
    # feature_csv_path_dict = {"customer": "data/customers.csv", "article": "data/articles.csv"}
    # dataset = MyGraphDataset(csv_file_path, root=args.root, num_rows=args.num_rows,
    #                         train_category="Sweater",
    #                         test_category=args.test_category,
    #                         dataset_name="hm", mode=args.mode)
    # data = dataset[0]
    # print(data)

# More testing on the correctness, stats of the graphs created
# # Assuming you have edge_index, edge_attr, and num_nodes already defined
# edge_index = data.edge_index
# edge_attr = data.edge_attr
# num_nodes = data.num_nodes

# # Step 2: Calculate basic statistics
# deg = degree(edge_index[0], num_nodes=num_nodes)
# avg_degree = deg.mean().item()
# density = (edge_index.shape[1]) / (num_nodes * (num_nodes - 1))

# print(f"Average degree: {avg_degree}")
# print(f"Density: {density}")

# # Identify the node with the highest degree
# highest_degree, high_degree_node = deg.max(0)

# G = to_networkx(data, to_undirected=True)

# # find the connected components of G
# connected_components = list(nx.connected_components(G))

# total_components = len(connected_components)
# sizes = [len(comp) for comp in connected_components]
# max_size = max(sizes)
# min_size = min(sizes)
# average_size = sum(sizes) / total_components

# print(f"Total Number of Connected Components: {total_components}")
# print(f"Maximum Size of Connected Components: {max_size}")
# print(f"Minimum Size of Connected Components: {min_size}")
# print(f"Average Size of Connected Components: {average_size:.2f}")

# up_data, up_user2node, up_product2node, feature_key = get_user_product_graph(
#     "2019-Nov.csv", 100
# )
# up_data, up_user2node, up_product2node = load_max_connected(
#     up_data, up_user2node, up_product2node
# )
# print(up_data, up_user2node, up_product2node)
# max_connected = dataset.load_max_connected()
# print(max_connected)
# 1-by-n vector (one-hot)
# matrix multiplication to get reachability (threshold: >0 => 1)
# reachable in one step
# matrix multiplcation with the adj matrix again and again
# repeat until convergence
# subset the adj matrix with this indices