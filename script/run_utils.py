from ast import literal_eval
import argparse
from pathlib import Path
import yaml
import jinja2
import subprocess
import shlex
import os

import pandas as pd


def output2dataframe(output_dir: Path):

    row = {
        "exp_id": output_dir.name,
        "date": output_dir.parent.name,
        "dataset_name": output_dir.parent.parent.name,
        "model": output_dir.parent.parent.parent.name,
    }

    output_file = output_dir / "log.txt"
    lines = open(output_file, "r").readlines()

    row["hash"] = lines[1].split()[-1]
    row["seed"] = lines[2].split()[-1]
    row["train_categories"] = (
        lines[3].split(":")[-1].strip().replace(",", "").replace("'", "")
    )

    # Clean up the test categories line before passing it to literal_eval
    test_categories_str = lines[4].split(":")[-1].strip()
    print(f"Raw test categories string: {test_categories_str}")  # Debug statement

    try:
        test_categories = literal_eval(test_categories_str)
        print(f"Parsed test categories: {test_categories}")  # Debug statement
    except (SyntaxError, ValueError) as e:
        print(f"Error in literal_eval: {e}")
        print(f"Attempted to parse: {test_categories_str}")
        raise
    
    index = [i for i, line in enumerate(lines) if "Evaluate on test" in line][0]

    rows = []
    start_category = index
    for i, category in enumerate(test_categories):
        row["test_category"] = category

        row["mr"] = lines[start_category + 1].split()[-1]
        row["mrr"] = lines[start_category + 2].split()[-1]
        row["hits@1"] = lines[start_category + 3].split()[-1]
        row["hits@3"] = lines[start_category + 4].split()[-1]
        row["hits@10"] = lines[start_category + 5].split()[-1]
        row["hits@10_50"] = lines[start_category + 6].split()[-1]

        rows.append(row.copy())
        start_category += 6

    new_df = pd.DataFrame.from_dict(rows)

    output_csv = Path("results.csv")
    if output_csv.exists():
        df = pd.read_csv(output_csv)
    else:
        df = pd.DataFrame(columns=rows[0].keys())

    pd.concat([df, new_df]).to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seeds", nargs="*")
    parser.add_argument("--train_categories_tuples", nargs="*")
    parser.add_argument("--test_categories", nargs="*")
    parser.add_argument("--feature_methods", nargs="*")

    args = parser.parse_args()

    with open(args.config, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render({})
    cfg = yaml.safe_load(instance)

    if args.feature_methods is None:
        args.feature_methods = ["ours"]

    for feature_method in args.feature_methods:
        cfg["model"]["feature_method"] = feature_method
        for i, train_categories in enumerate(args.train_categories_tuples):
            train_categories = train_categories.strip("()").split(",")
            for seed in args.seeds:
                cfg["dataset"]["train_categories"] = train_categories
                # if test_categories is not provided, use the remaining from [(computers.desktop), (appliances.kitchen.refrigerators), (furniture.bedroom.bed), (electronics.smartphone), (apparel.shoes)]
                if args.test_categories is not None:
                    cfg["dataset"]["test_categories"] = args.test_categories
                else:
                    cfg["dataset"]["test_categories"] = [
                        x
                        for x in [
                            "computers.desktop",
                            "appliances.kitchen.refrigerators",
                            "furniture.bedroom.bed",
                            "electronics.smartphone",
                            "apparel.shoes",
                        ]
                        if x not in train_categories
                    ]

                yaml_file = f"config/inductive/baseline{str(i)}.yaml"
                with open(yaml_file, "w") as ff:
                    yaml.dump(cfg, ff, allow_unicode=True)

                output = subprocess.run(
                    shlex.split(
                        f"python script/run.py -c {yaml_file} --gpus [0] --version v1 -s {seed}",
                    ),
                    env=os.environ,
                    stdout=subprocess.PIPE,
                )

                output_dir = Path(str(output.stdout).split("\\n")[0].split(" ")[2])

                output2dataframe(output_dir)


if __name__ == "__main__":
    main()
