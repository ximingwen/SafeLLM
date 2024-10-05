# 完整运行

base_model_path='meta-llama/Meta-Llama-3-8B'
deepspeed_config_name=./config/ds.json
output_path='./output'

model_pretrained_lora_path=${output_path}'/pretrained_lora'
model_pretrained_full_path=${output_path}'/pretrained_full'
model_sft_lora_path=${output_path}'/sft_lora'
model_sft_full_path=${output_path}'/sft_full'
model_reward_model_lora_path=${output_path}'/reward_model_lora'
model_ppo_lora_path=${output_path}'/ppo_lora'
model_ppo_full_path=${output_path}'/ppo_full'
model_dpo_lora_path=${output_path}'/dpo_lora'
model_dpo_full_path=${output_path}'/dpo_full'


echo '-------------------------------------------------------'
date
echo '-------------------------------------------------------'

# stage dpo
rm_dataset_name='Anthropic/hh-rlhf'
deepspeed ./ma-rlhf/dpo.py \
	--dataset_name=${rm_dataset_name} \
	--model_name=${model_sft_full_path} \
	--output_name=${model_dpo_lora_path} \
	--use_QLora=True \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--batch_size=16 \
	--num_train_epochs=1 \
	--seq_length=512 \
	--gradient_accumulation_steps=4 \
	--learning_rate=2e-5

 # merge DPO
 python ./ma-rlhf/merge_adapter.py \
 	--base_model_name=${model_sft_full_path} \
 	--model_name=${model_dpo_lora_path} \
 	--merged_model_name=${model_dpo_full_path}

bash ./scripts/run_generation_examples.sh ${model_dpo_full_path} 512

#  # generate result
#  echo "------------------print dpo result------------------"
#  python ./ma-rlhf/generate.py \
#  	--model_name=${model_dpo_full_path} \
#  	--prompt='give me a C++ code about quick sort.' \
#  	--max_new_tokens=512
