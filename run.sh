# run the following command for 3 different seeds
# python script/run.py --config_path config/inductive/baseline_new.yaml --use_wb 1 --save_dir exp/sample/run2 --seed 2

# for seed in 4 5 6
# do
#     python script/run.py --config_path config/inductive/baseline_new.yaml --use_wb 1 --save_dir exp/sample/run3 --seed $seed
# done

# sweep over different --edgegraph.rgcn.num_bases (3 5 10) and --edgegraph.rgcn.cat_method (cat, add)
# name save_dir as exp/sample/rgcn_numbases(num_bases)_catmethod(cat_method)
for num_bases in 1 2
do
    for cat_method in cat add
    do
        python script/run.py --config_path config/inductive/baseline_new.yaml --use_wb 1 --save_dir exp/sample/rgcn_numbases${num_bases}_catmethod${cat_method} --edgegraph.rgcn.num_bases $num_bases --edgegraph.rgcn.cat_method $cat_method --seed 1
    done
done

# for num_bases in 0
# do
#     for cat_method in cat add
#     do
#         python script/run.py --config_path config/inductive/baseline_new.yaml --use_wb 1 --save_dir exp/sample/rgcn_catmethod${cat_method} --edgegraph.rgcn.num_bases $num_bases --edgegraph.rgcn.cat_method $cat_method --seed 1
#     done
# done
