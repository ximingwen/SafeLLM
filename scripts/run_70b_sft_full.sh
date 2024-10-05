# 完整运行
DS_SKIP_CUDA_CHECK=1
base_model_path='meta-llama/Meta-Llama-3-70B'
deepspeed_config_name='./config/ds_full_offload.json'
output_path='./output'

model_pretrained_lora_path=${output_path}'/pretrained_lora'
model_pretrained_full_path=${output_path}'/pretrained_full'
model_sft_lora_path=${output_path}'/sft_lora'
model_sft_full_path=${output_path}'/sft_full'


echo '-------------------------------------------------------'
date
echo '-------------------------------------------------------'

# # stage: sft
sft_dataset_name='yahma/alpaca-cleaned'
model_pretrained_full_path=${base_model_path}

accelerate launch --config_file ./config/deepspeed_zero3.yaml ./ma-rlhf/sft_70b.py \
	--dataset_name=${sft_dataset_name} \
	--model_name=${model_pretrained_full_path} \
	--seq_length=128 \
	--output_name=${model_sft_full_path} \
	--use_QLora=False \
	--batch_size=1 \
	--use_flash_attention_2=False \
	--num_train_epochs=1 \
	--gradient_accumulation_steps=1 \
	--learning_rate=1e-5 \
	--num_gpus=8 \
	--num_nodes=1

# 全参微调不需要保存adapter
# bash ./scripts/run_generation_examples.sh ${model_sft_full_path} 512
