import argparse
import os
from shutil import copyfile

import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model


def merge_lora(lora_path, device_map=None, base_path="./chatglm2-6b"):
    """合并lora模型和base模型"""
    if device_map is None:
        device_map = {'': 'cpu'}
    config = PeftConfig.from_pretrained(lora_path)
    base_model = AutoModel.from_pretrained(base_path, trust_remote_code=True, torch_dtype=torch.float32,
                                           device_map=device_map)
    model = PeftModel.from_pretrained(base_model, lora_path, device_map=device_map)
    model = model.merge_and_unload()
    return model, config




def save_model_and_tokenizer(model, base_model_path, output_path, base_path):
    """保存模型和tokenizer相关配置"""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, max_shard_size="1GB")
    for fp in os.listdir(base_path):  # 拷贝remote_scripts中的官方脚本到最终输出的文件夹中，供load模型时使用
        if fp.split('.')[-1] == 'py':
            copyfile(os.path.join(base_path, fp),
                     os.path.join(output_path, fp))


def main(lora_path, output_path, base_path, device_map=None):
    if device_map is None:
        device_map = {'': 'cpu'}
    merged_model, lora_config = merge_lora(lora_path, device_map, base_path=base_path)
    save_model_and_tokenizer(merged_model, base_path, output_path, base_path)
    print(f'merge成功！')


def parse_args():
    parser = argparse.ArgumentParser(description='ChatGLM2-6B merge lora.')
    parser.add_argument('--lora_path', type=str, required=True, help='LoRA训练后保存模型的目录')
    parser.add_argument('--output_path', type=str, default='./merge_model', help='最终保存合并后的模型目录')
    parser.add_argument('--device', type=str, default='auto', help='device_map')
    parser.add_argument('--base_path', type=str, default='./chatglm2-6b', help='基座模型目录')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.device != 'auto':
        device_map = {'': args.device}
    else:
        device_map = 'auto'
    main(lora_path=args.lora_path,
         output_path=args.output_path,
         base_path=args.base_path,
         device_map=device_map)