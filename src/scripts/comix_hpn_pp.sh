!/bin/sh
seed_max=5


for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../main.py --config=comix_hpn_pp --env-config=particle with env_args.scenario_name=continuous_pred_prey_3a t_max=2000000 seed=${seed} use_wandb=True
done