./scripts/run_7b_sft_qlora.sh > ./output/log_sft_qlora.txt 2>&1 &

./scripts/run_7b_dpo.sh > ./output/log_dpo.txt 2>&1 &

./scripts/run_7b_ppo.sh > ./output/log_ppo.txt 2>&1 &
