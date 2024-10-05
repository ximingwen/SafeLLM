deepspeed_config_name='./config/ds.json'
output_path='./output'
sft_path='./output/dpo_full'

model_sft_full_path=${sft_path}
model_reward_model_lora_path=${output_path}'/reward_model_lora'
model_ppo_lora_path=${output_path}'/ppo_lora'
model_ppo_full_path=${output_path}'/ppo_full'

echo '-------------------------------------------------------'
date
echo '-------------------------------------------------------'

echo 'dpo model has trained by ./scripts/run_all_7b_dpo.sh'

# 如果有更多计算资源 建议使用 `PKU-Alignment/PKU-SafeRLHF-30K` 数据集

# stage reward model
rm_dataset_name='PKU-Alignment/PKU-SafeRLHF-10K'
deepspeed ./ma-rlhf/reward_model.py \
	--dataset_name=${rm_dataset_name} \
	--model_name=${model_sft_full_path} \
	--seq_length=512 \
	--batch_size=16 \
	--output_name=${model_reward_model_lora_path} \
	--use_QLora=True \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--num_train_epochs=2 \
	--gradient_accumulation_steps=2 \
	--learning_rate=2e-5

test reward model
python test/test_reward.py


#  echo '-------------------------------------------------------'
#  date
#  echo '-------------------------------------------------------'

# stage ppo prior using DPO-Model as base model
# ~ 4hour with 8 * 3090-24GB
rm_dataset_name='PKU-Alignment/PKU-SafeRLHF-10K'
deepspeed  --num_gpus 8  ./ma-rlhf/ppo.py \
	--dataset_name=${rm_dataset_name} \
	--model_name=${model_sft_full_path} \
	--reward_model_name=${model_reward_model_lora_path} \
	--output_name=${model_ppo_lora_path} \
	--use_QLora=True \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--batch_size=16 \
	--mini_batch_size=1 \
	--ppo_epochs=1 \
	--output_max_length=256 \
	--seq_length=64 \
	--gradient_accumulation_steps=2

# # merge PPO
python ./ma-rlhf/merge_adapter.py \
	--base_model_name=${model_sft_full_path} \
	--model_name=${model_ppo_lora_path} \
	--merged_model_name=${model_ppo_full_path}

bash ./scripts/run_generation_examples.sh ${model_ppo_full_path} 512


#  # generate result
#  echo "------------------print ppo result------------------"
#  python ./ma-rlhf/generate.py \
#  	--model_name=${model_ppo_full_path} \
#  	--prompt='give me a C++ code about quick sort.' \
#  	--max_new_tokens=512
