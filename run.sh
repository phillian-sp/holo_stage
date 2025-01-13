# sweep over different configs --config_path config/inductive/baseline_phone.yaml config/inductive/baseline_refrig.yaml config/inductive/baseline_shoe.yaml config/inductive/baseline_bed.yaml
# name save_dir as exp/sample/rgcn_config(config)
# for config in phone refrig shoe bed
# do
#     for method in method1 method2 method3 method4
#     do
#         python script/run.py --config_path config/${config}.yaml --use_wb 1 --save_dir exp/final_${config}/compgcn_${method} --edgegraph.compgcn.edge_method $method --seed 1 --edgegraph.final_model compgcn
#         python script/run.py --config_path config/${config}.yaml --use_wb 1 --save_dir exp/final_${config}/rgcn_${method} --edgegraph.compgcn.edge_method $method --seed 1 --edgegraph.final_model rgcn
#     done
#     python script/run.py --config_path config/${config}.yaml --use_wb 1 --save_dir exp/final_${config}/nbf --edgegraph.final_model nbf --seed 1  --edgegraph.final_model nbf
# done

# for config in refrig
# do
#     for method in method1 method2 method3 method4
#     do
#         python script/run.py --config_path config/${config}.yaml --use_wb 1 --save_dir exp/comp_prob/compgcn_${config}_${method} --edgegraph.compgcn.edge_method $method --seed 1 --edgegraph.final_model compgcn
#     done
# done

for config in refrig
do
    python script/run.py --config_path config/${config}.yaml --use_wb 1 --save_dir exp/comp_prob/compgcn_${config}_${method} --seed 1 --edgegraph.final_model compgcn --edgegraph.compgcn.use_stage 0
done
