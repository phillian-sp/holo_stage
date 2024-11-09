# run the following command for 3 different seeds
# python script/run.py --config_path config/inductive/baseline_new.yaml --use_wb 1 --save_dir exp/sample/run2 --seed 2

for seed in 4 5 6
do
    python script/run.py --config_path config/inductive/baseline_new.yaml --use_wb 1 --save_dir exp/sample/run3 --seed $seed
done
