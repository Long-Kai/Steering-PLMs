### Steering PLMs

Code for ICML'25 paper "Steering Protein Language Models". 

## Process

### Extract steering vector

#### ESM2-650M

```shell
python3 extract_esm2_steering_vec.py --model "650M" --num_data 100 --property "therm" --data_path "data/therm_filtered.csv" --theshold_pos 70.0 --theshold_neg 50.0
python3 extract_esm2_steering_vec.py --model "650M" --num_data 100 --property "sol" --data_path "data/sol_filtered.csv" --theshold_pos 0.5 --theshold_neg 0.2
```

#### ESM3-open

```shell
python3 extract_esm3_steering_vec.py --num_data 100 --property "therm" --data_path "data/therm_filtered.csv" --theshold_pos 70.0 --theshold_neg 50.0
python3 extract_esm3_steering_vec.py --num_data 100 --property "sol" --data_path "data/sol_filtered.csv" --theshold_pos 0.5 --theshold_neg 0.2
```

#### ProLLaMA

```shell
python3 extract_prollama_steering_vec.py --model_dir "../RLHF/Edit_GPT/checkpoint_prollama" --num_data 100 --property "therm" --data_path "data/therm_filtered.csv" --theshold_pos 70.0 --theshold_neg 50.0

python3 extract_prollama_steering_vec.py --model_dir "../RLHF/Edit_GPT/checkpoint_prollama" --num_data 100 --property "sol" --data_path "data/sol_filtered.csv" --theshold_pos 0.5 --theshold_neg 0.2
```

### Generation

#### ESM2-650M

```shell
python3 steering_esm2_generation.py --model "650M" --property "therm" --ref_data_path "data/therm_easy.csv" --output_file "results/ESM2_gen_steering_therm_easy.csv" --steering --n 100
python3 steering_esm2_generation.py --model "650M" --property "therm" --ref_data_path "data/therm_hard.csv" --output_file "results/ESM2_gen_steering_therm_hard.csv" --steering --n 100
python3 steering_esm2_generation.py --model "650M" --property "sol" --ref_data_path "data/sol_easy.csv" --output_file "results/ESM2_gen_steering_sol_easy.csv" --steering --n 100
python3 steering_esm2_generation.py --model "650M" --property "sol" --ref_data_path "data/sol_hard.csv" --output_file "results/ESM2_gen_steering_sol_hard.csv" --steering --n 100
```

#### ESM3-open

```shell
python3 steering_esm3_generation.py --property "therm" --ref_data_path "data/therm_easy.csv" --output_file "results/ESM3_gen_steering_therm_easy.csv" --steering --n 100
python3 steering_esm3_generation.py --property "therm" --ref_data_path "data/therm_hard.csv" --output_file "results/ESM3_gen_steering_therm_hard.csv" --steering --n 100
python3 steering_esm3_generation.py --property "sol" --ref_data_path "data/sol_easy.csv" --output_file "results/ESM3_gen_steering_sol_easy.csv" --steering --n 100
python3 steering_esm3_generation.py --property "sol" --ref_data_path "data/sol_hard.csv" --output_file "results/ESM3_gen_steering_sol_hard.csv" --steering --n 100
```

#### ProLLaMA

```shell
python3 steering_prollama_generation.py --model_dir [dir_to_checkpoint] --property "therm" --output_file "results/prollama_gen_steering_therm.csv" --steering --steer_only_first_token --n 100
python3 steering_prollama_generation.py --model_dir [dir_to_checkpoint] --property "sol" --output_file "results/prollama_gen_steering_sol.csv" --steering --n 100 
```

### Optimization

#### ESM3-open

```shell
python3 steering_esm3_optimization.py --property "therm" --data_path "data/therm_easy.csv" --output_file "results/ESM3_opt_therm_easy.csv"  --n 100 --round 2 --T 8
python3 steering_esm3_optimization.py --property "therm" --data_path "data/therm_hard.csv" --output_file "results/ESM3_opt_therm_hard.csv"  --n 100 --round 6 --T 8
python3 steering_esm3_optimization.py --property "sol" --data_path "data/sol_easy.csv" --output_file "results/ESM3_opt_sol_easy.csv"  --n 100 --round 2 --T 2
python3 steering_esm3_optimization.py --property "sol" --data_path "data/sol_hard.csv" --output_file "results/ESM3_opt_sol_hard.csv"  --n 100 --round 8 --T 2
```

## Citation

If you find this code useful for your research, please cite our paper:

```
@inproceedings{huang2025steering,
  title={Steering Protein Language Models},
  author={Huang, Long-Kai and Zhu, Rongyi and He, Bing and Yao, Jianhua},
  booktitle={International Conference on Machine Learning},
  year={2025},
  organization={PMLR}
}
```