import torch
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.model_lora import apply_lora, load_lora

def merge_lora_with_base(base_model_path, lora_path, output_path, lora_rank=16):
    # 初始化模型配置（根据实际情况调整参数）
    config = LMConfig(dim=512, n_layers=8, max_seq_len=8192, use_moe=False)
    
    # 加载基础模型
    model = MiniMindLM(config)
    model.load_state_dict(torch.load(base_model_path, map_location='cpu'))
    
    # 应用LoRA结构
    apply_lora(model, rank=lora_rank)
    
    # 加载LoRA权重
    load_lora(model, lora_path)
    
    # 合并LoRA权重到基础模型
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 获取LoRA的A、B矩阵
            A = module.lora.A.weight.data
            B = module.lora.B.weight.data
            
            # 计算增量权重 ΔW = B @ A
            delta_W = torch.mm(B, A)
            
            # 更新原始权重
            module.weight.data += delta_W
            
            # 移除LoRA相关属性和恢复原forward
            del module.lora
            module.forward = torch.nn.Linear.forward.__get__(module, torch.nn.Linear)
    
    # 保存合并后的模型
    torch.save(model.state_dict(), output_path)
    print(f"模型已成功合并并保存至 {output_path}")

if __name__ == "__main__":
    base_model_path = "out/rlhf_512.pth"
    lora_model_path = "out/lora/lora_identity_512.pth"  # 替换为实际路径
    merged_model_path = "out/merged_model.pth"
    
    merge_lora_with_base(base_model_path, lora_model_path, merged_model_path)