# 完整运行

base_model_path='meta-llama/Meta-Llama-3-8B'
# base_model_path='./output/sft_lora'
deepspeed_config_name=./config/ds.json
output_path='./output'

model_pretrained_lora_path=${output_path}'/pretrained_lora'
model_pretrained_full_path=${output_path}'/pretrained_full'
model_sft_lora_path=${output_path}'/sft_lora'
model_sft_full_path=${output_path}'/sft_full'

echo '-------------------------------------------------------'
date
echo '-------------------------------------------------------'

# # stage: sft
# sft_dataset_name='yahma/alpaca-cleaned'
# sft_dataset_name='vicgalle/alpaca-gpt4'
sft_dataset_name='xiaodongguaAIGC/alpaca_en_zh_ruozhiba'
model_pretrained_full_path=${base_model_path}
deepspeed ./ma-rlhf/sft_pack.py \
	--dataset_name=${sft_dataset_name} \
	--model_name=${model_pretrained_full_path} \
	--seq_length=512 \
	--output_name=${model_sft_lora_path} \
	--use_QLora=True \
	--batch_size=16 \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--num_train_epochs=2 \
	--gradient_accumulation_steps=1 \
	--learning_rate=2e-5

# merge SFT
python ./ma-rlhf/merge_adapter.py \
	--base_model_name=${model_pretrained_full_path} \
	--model_name=${model_sft_lora_path} \
	--merged_model_name=${model_sft_full_path}

bash ./scripts/run_generation_examples.sh ${model_sft_full_path} 512
