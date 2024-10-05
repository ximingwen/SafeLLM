# 完整运行
DS_SKIP_CUDA_CHECK=1
base_model_path='meta-llama/Meta-Llama-3-8B'
deepspeed_config_name=./config/ds_full.json
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
sft_dataset_name='xiaodongguaAIGC/alpaca_en_zh_ruozhiba'
model_pretrained_full_path=${base_model_path}
deepspeed ./ma-rlhf/sft.py \
	--dataset_name=${sft_dataset_name} \
	--model_name=${model_pretrained_full_path} \
	--seq_length=512 \
	--output_name=${model_sft_full_path} \
	--use_QLora=False \
	--batch_size=4 \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--num_train_epochs=1 \
	--gradient_accumulation_steps=4 \
	--learning_rate=1e-5

# 全参微调不需要保存adapter
bash ./scripts/run_generation_examples.sh ${model_sft_full_path} 512

#  # generate result
# # model_sft_full_path=${model_sft_lora_path}
#  echo "------------------print sft result------------------"
#  python ./ma-rlhf/generate.py \
#  	--model_name=${model_sft_full_path} \
#  	--prompt='give me a C++ code about quick sort.' \
#  	--max_new_tokens=1024
