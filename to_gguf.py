import torch
import gguf
from transformers import AutoTokenizer
from model.model import MiniMindLM
from model.LMConfig import LMConfig

def convert_pth_to_gguf(
    pth_path: str,
    tokenizer_dir: str,
    gguf_path: str,
    model_name: str = "meuai",
    dtype: gguf.GGMLQuantizationType = gguf.GGMLQuantizationType.F32
):
    # 加载模型配置（需与实际训练配置一致）
    config = LMConfig(
        dim=512,
        n_layers=8,
        n_heads=8,
        vocab_size=6400,
        max_seq_len=8192,
        norm_eps=1e-5,
        rope_theta=1e6,
    )
    
    # 加载PyTorch模型
    model = MiniMindLM(config)
    model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    model.eval()

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # 初始化GGUF写入器
    gguf_writer = gguf.GGUFWriter(gguf_path, model_name)

    # 添加模型元数据
    gguf_writer.add_uint32('n_vocab', config.vocab_size)
    gguf_writer.add_uint32('n_embd', config.dim)
    gguf_writer.add_uint32('n_head', config.n_heads)
    gguf_writer.add_uint32('n_layer', config.n_layers)
    gguf_writer.add_float32('layer_norm_epsilon', config.norm_eps)
    gguf_writer.add_uint32('n_ctx', config.max_seq_len)

    # 处理tokenizer
    tokenizer_config = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }

    # 添加特殊token
    gguf_writer.add_bos_token_id(tokenizer.bos_token_id)
    gguf_writer.add_eos_token_id(tokenizer.eos_token_id)
    gguf_writer.add_pad_token_id(tokenizer.pad_token_id)
    gguf_writer.add_unk_token_id(tokenizer.unk_token_id)

    # 添加常规token
    tokens = []
    scores = []
    toktypes = []

    for idx in range(config.vocab_size):
        token = tokenizer.convert_ids_to_tokens(idx)
        toktype = gguf.TokenType.NORMAL

        if token in tokenizer.all_special_tokens:
            toktype = gguf.TokenType.CONTROL
        elif token.startswith("<0x") and token.endswith(">"):
            toktype = gguf.TokenType.BYTE
        
        tokens.append(token.encode("utf-8", errors="replace"))
        scores.append(-idx)  # 默认分数
        toktypes.append(toktype)

    gguf_writer.add_tokenizer_model("llama")  # 假设使用类似LLaMA的tokenizer
    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)
    gguf_writer.add_token_types(toktypes)

    # 写入模型权重
    def add_tensor(name, tensor):
        data = tensor.squeeze().float().detach().numpy()
        gguf_writer.add_tensor(
            name=name,
            tensor=data,
            raw_shape=data.shape,
            # dtype=dtype_bit,
            raw_dtype=gguf.GGMLQuantizationType.F32
        )

    # 嵌入层
    add_tensor("tok_embeddings.weight", model.tok_embeddings.weight)

    # 各Transformer层
    for layer_idx in range(config.n_layers):
        layer = model.layers[layer_idx]
        prefix = f"layers.{layer_idx}"

        # 注意力层
        add_tensor(f"{prefix}.attention.wq.weight", layer.attention.wq.weight.T)
        add_tensor(f"{prefix}.attention.wk.weight", layer.attention.wk.weight.T)
        add_tensor(f"{prefix}.attention.wv.weight", layer.attention.wv.weight.T)
        add_tensor(f"{prefix}.attention.wo.weight", layer.attention.wo.weight.T)

        # 前馈网络
        add_tensor(f"{prefix}.feed_forward.w1.weight", layer.feed_forward.w1.weight.T)
        add_tensor(f"{prefix}.feed_forward.w2.weight", layer.feed_forward.w2.weight.T)
        add_tensor(f"{prefix}.feed_forward.w3.weight", layer.feed_forward.w3.weight.T)

        # 归一化层
        add_tensor(f"{prefix}.attention_norm.weight", layer.attention_norm.weight)
        add_tensor(f"{prefix}.ffn_norm.weight", layer.ffn_norm.weight)

    # 最终归一化层
    add_tensor("norm.weight", model.norm.weight)

    # 输出层
    add_tensor("output.weight", model.output.weight)

    # 完成写入
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    print(f"成功生成GGUF文件：{gguf_path}")

if __name__ == "__main__":
    convert_pth_to_gguf(
        pth_path="out/merged_model.pth",
        tokenizer_dir="./model/minimind_tokenizer",
        gguf_path="out/meuai.gguf"
    )