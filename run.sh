# run the following command for 3 different seeds
# python script/run.py --config_path config/inductive/baseline_new.yaml --use_wb 1 --save_dir exp/sample/run2 --seed 2

# for seed in 4 5 6
# do
#     python script/run.py --config_path config/inductive/baseline_new.yaml --use_wb 1 --save_dir exp/sample/run3 --seed $seed
# done

# sweep over different --edgegraph.rgcn.num_bases (3 5 10) and --edgegraph.rgcn.cat_method (cat, add)
# name save_dir as exp/sample/rgcn_numbases(num_bases)_catmethod(cat_method)
# for num_bases in 1 2
# do
#     for cat_method in cat add
#     do
#         python script/run.py --config_path config/inductive/baseline_new.yaml --use_wb 1 --save_dir exp/sample/rgcn_numbases${num_bases}_catmethod${cat_method} --edgegraph.rgcn.num_bases $num_bases --edgegraph.rgcn.cat_method $cat_method --seed 1
#     done
# done

# for num_bases in 0
# do
#     for cat_method in cat add
#     do
#         python script/run.py --config_path config/inductive/baseline_new.yaml --use_wb 1 --save_dir exp/sample/rgcn_catmethod${cat_method} --edgegraph.rgcn.num_bases $num_bases --edgegraph.rgcn.cat_method $cat_method --seed 1
#     done
# done

# sweep over different --edgegraph.rgcn.num_layers (1, 3, 5) and --lr (5.0e-2, 5.0e-3, 5.0e-4, 5.0e-5)
# name save_dir as exp/sample/rgcn_numlayers(num_layers)_lr(lr)
# for num_layers in 6
# do
#     for lr in 5.0e-2 5.0e-3 5.0e-4 5.0e-5
#     do
#         python script/run.py --config_path config/inductive/baseline_new.yaml --use_wb 1 --save_dir exp/sample/rgcn_numlayers${num_layers}_lr${lr} --edgegraph.rgcn.num_layers $num_layers --lr $lr --seed 1
#     done
# done

# sweep over different configs --config_path config/inductive/baseline_phone.yaml config/inductive/baseline_refrig.yaml config/inductive/baseline_shoe.yaml config/inductive/baseline_bed.yaml
# name save_dir as exp/sample/rgcn_config(config)
for config in phone refrig shoe bed
do
    for method in method1 method2 method3 method4
    do
        python script/run.py --config_path config/${config}.yaml --use_wb 1 --save_dir exp/final_${config}/compgcn_${method} --edgegraph.compgcn.edge_method $method --seed 1 --edgegraph.final_model compgcn
        python script/run.py --config_path config/${config}.yaml --use_wb 1 --save_dir exp/final_${config}/rgcn_${method} --edgegraph.compgcn.edge_method $method --seed 1 --edgegraph.final_model rgcn
    done
    python script/run.py --config_path config/${config}.yaml --use_wb 1 --save_dir exp/final_${config}/nbf --edgegraph.final_model nbf --seed 1  --edgegraph.final_model nbf
done
