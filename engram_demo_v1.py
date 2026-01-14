"""
================================================================================
[Engram 架构演示实现 - 带详细中文注释版]

免责声明:
1. 仅供演示: 
   本代码仅用于展示 Engram 模块的核心逻辑和数据流，并非生产级代码。
2. 生产环境需优化: 
   实际生产使用需要定制 CUDA 内核（用于极速哈希查表）和分布式训练支持。
3. 简化部分: 
   为了专注于 Engram 模块，标准的 Transformer 组件（如 Attention, MoE）和复杂的
   Hyper-connection (mHC) 机制在此处仅做了模拟（Mock）。
================================================================================
"""

"""
依赖安装:
pip install torch numpy transformers sympy
"""

## 内置库
from typing import List
from dataclasses import dataclass, field
import math

## 第三方库
from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex 

# =============================================================================
# 1. 配置类 (Configuration)
# =============================================================================

@dataclass
class EngramConfig:
    """Engram 模块的超参数配置"""
    # 使用的分词器，DeepSeek-V3 的词表很大 (128k)
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    # Engram 每一层的词表大小预算（不同 N-gram 共享这个预算）
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280*5, 129280*5])
    # 最大 N-gram 长度（例如 3 表示同时检索 2-gram 和 3-gram）
    max_ngram_size: int = 3
    # 每个 N-gram 检索出的向量总维度
    n_embed_per_ngram: int = 512
    # 哈希头的数量（为了减少哈希冲突，使用多头哈希）
    n_head_per_ngram: int = 8
    # 在 Transformer 的第几层插入 Engram 模块（论文建议在浅层，如第 1 层和 15 层）
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    # 融合后的卷积核大小
    kernel_size: int = 4
    
@dataclass
class BackBoneConfig:
    """主干模型 (Transformer) 的配置"""
    hidden_size: int = 1024
    # DeepSeek-V3 特有的 mHC (Manifold-Constrained Hyper-Connections) 扩展倍数
    # 可以理解为隐状态被扩展了 hc_mult 倍的并行分支
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30
    
engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()

# =============================================================================
# 2. 词表压缩 (Tokenizer Compression)
# 对应论文 Section 2.2: Tokenizer Compression
# =============================================================================

class CompressedTokenizer:
    """
    压缩分词器：将语义相同的不同 Token 映射到同一个 ID。
    例如：' Apple' (带空格) 和 'Apple' (不带空格) 可能被映射为同一个 ID。
    目的：提高 N-gram 表的语义密度，减少存储浪费。
    """
    def __init__(
        self,
        tokenizer_name_or_path,
    ):
        # 加载原始分词器
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        
        # 定义归一化规则：NFKC标准化、去重音、转小写、合并空格等
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL), # 处理纯空格
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        
        # 构建映射表：Raw Token ID -> Canonical (Compressed) Token ID
        self.lookup_table, self.num_new_token = self._build_lookup_table()
    
    def __len__(self):
        return self.num_new_token
    
    def _build_lookup_table(self):
        """构建旧ID到新ID的映射表"""
        old2new = {}
        key2new = {}          
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            # 解码当前 token
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            
            # 如果包含乱码/特殊字符，保持原样；否则进行归一化
            if "" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            # 如果归一化后的 key 已经存在，则复用其 ID
            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        
        # 创建 numpy 数组以便快速索引
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)
    
    def _compress(self, input_ids):
        """将原始 input_ids 转换为压缩后的 IDs"""
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        # 查表替换
        out[pos_mask] = self.lookup_table[valid_ids]
        return out   
    
    def __call__(self, input_ids):
        return self._compress(input_ids)

# =============================================================================
# 3. 短卷积模块 (Short Convolution)
# 对应论文 Section 2.3 结尾部分
# =============================================================================

class ShortConv(nn.Module):
    """
    轻量级深度卷积层。
    作用：融合 Engram 检索出来的特征，增加局部非线性，扩大感受野。
    """
    def __init__(
        self, 
        hidden_size: int, 
        kernel_size: int = 4, 
        dilation: int = 1, 
        norm_eps: float = 1e-5,
        hc_mult: int = 4, # mHC 分支数
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation
        
        # 输入通道总数 = 隐藏层维度 * 分支数
        total_channels = hidden_size * hc_mult
        
        # 深度卷积 (Depthwise Conv): groups=total_channels
        # 这意味着每个通道独立进行卷积，参数量极小
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation, # 因果卷积 padding (Causal Padding)
            dilation=dilation,
        )

        # 每个分支独立的 RMSNorm
        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps) 
            for _ in range(hc_mult)
        ])
        
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (Batch, Length, HC_MULT, Hidden_Dim)
        Output: 同 Input
        """
        B, T, G, C = x.shape
        
        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        # 1. 对每个 mHC 分支分别做 Norm
        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))
        
        # 2. 拼接并在时间维度做卷积
        x_norm = torch.cat(normed_chunks, dim=-1) # (B, T, G*C)
        x_bct = x_norm.transpose(1, 2)            # 转置为 (B, G*C, T) 供 Conv1d 使用
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]                    # 裁剪掉多余的 padding，保证因果性

        if self.activation:
            y_bct = self.act_fn(y_bct)
        
        # 3. 还原形状
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        
        return y
    
def find_next_prime(start, seen_primes):
    """辅助函数：寻找下一个未使用的质数，用于哈希表大小"""
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1

# =============================================================================
# 4. N-gram 哈希映射核心 (Hashing Logic)
# 对应论文 Section 2.2: Multi-Head Hashing
# =============================================================================

class NgramHashMapping:
    """
    负责将输入 Token 序列转换为 N-gram 的哈希索引。
    这是 Engram 能够进行 O(1) 查找的关键。
    """
    def __init__(
        self, 
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,  
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        # 初始化压缩分词器
        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            # Pad ID 也要映射到压缩空间
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        # 初始化哈希乘数 (Multipliers)，用于 Poly-Hash
        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        self.layer_multipliers = {}

        # 为每一层、每一个 N-gram 位置生成随机的哈希系数
        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64
            )
            multipliers = r * 2 + 1 # 保证是奇数
            self.layer_multipliers[layer_id] = multipliers

        # 计算每一层、每个 Head 的哈希表大小（使用质数以减少冲突）
        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        """为每个哈希头分配一个质数大小的词表空间"""
        seen_primes = set()
        vocab_size_across_layers = {}
        
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            # 从 2-gram 开始到 max_ngram_size
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1
                
                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
            
        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        """
        核心哈希函数：计算滑动窗口的 N-gram 哈希值。
        """
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        # 辅助函数：将序列向右平移 k 位，实现获取前文 Token
        def shift_k(k: int) -> np.ndarray:
            if k == 0: return x
            # Padding 在左侧（过去的时间步）
            shifted = np.pad(x, ((0, 0), (k, 0)),
                                mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        # 预先计算所有需要的平移版本
        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        
        # 遍历 N (例如 N=2, N=3)
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            # 取出构成 N-gram 的 n 个 token 序列
            tokens = base_shifts[:n]
            
            # 计算滚动哈希 (Multiplicative Hashing)
            # Hash = (t_0 * m_0) XOR (t_1 * m_1) ...
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            
            # 多头哈希：对同一个 N-gram，用不同的质数取模，得到多个索引
            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        
        # 堆叠结果: [B, T, Num_Ngrams * Num_Heads]
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        # 1. 压缩 Token IDs
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        # 2. 为每个 Engram 层计算哈希索引
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers

# =============================================================================
# 5. 多头嵌入表 (Storage)
# =============================================================================

class MultiHeadEmbedding(nn.Module):
    """
    高效存储：将所有哈希头的嵌入表合并为一个大的 Embedding 层。
    使用 offsets 来区分不同头在总表中的位置。
    """
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        
        # 计算偏移量
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        
        total_N = sum(list_of_N)
        # 这里的 total_N 对应 Engram-27B 中那个巨大的 5.7B 参数表
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T, Num_Heads]
        # 加上 offsets，使得每个头的索引指向总表中正确的段落
        shifted_input_ids = input_ids + self.offsets
        output = self.embedding(shifted_input_ids)
        return output
    
# =============================================================================
# 6. Engram 完整模块 (The Main Module)
# 对应论文 Section 2.1 Overview & 2.3 Context-aware Gating
# =============================================================================

class Engram(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        
        # 1. 实例化哈希映射器
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size = engram_cfg.max_ngram_size,
            n_embed_per_ngram = engram_cfg.n_embed_per_ngram,
            n_head_per_ngram = engram_cfg.n_head_per_ngram,
            layer_ids = engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id = engram_cfg.pad_id,
            seed = engram_cfg.seed,
        )
        
        # 2. 实例化物理存储 (Embedding Table)
        # 维度计算：单个哈希头的维度 = 总预算 / 头数
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y],
            D = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
        )
        
        # 3. 后处理卷积
        self.short_conv = ShortConv(
            hidden_size = backbone_config.hidden_size,
            kernel_size = engram_cfg.kernel_size,
            dilation    = engram_cfg.max_ngram_size,
            hc_mult     = backbone_config.hc_mult,
        )
        
        # 4. 门控机制的投影层 (Gating Projections)
        # 输入维度 = N-gram种类数(max-1) * 每种的维度
        engram_hidden_size = (engram_cfg.max_ngram_size-1) * engram_cfg.n_embed_per_ngram
        
        # Value 投影：将检索到的静态 embedding 投影到模型隐空间
        self.value_proj = nn.Linear(engram_hidden_size, backbone_config.hidden_size)
        
        # Key 投影：用于计算门控分数。注意每个 mHC 分支有独立的 Key 投影
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size, backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)]
        )
        
        # 归一化层，用于门控计算的稳定性
        self.norm1 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
    
    def forward(self, hidden_states, input_ids):
        """
        前向传播逻辑
        hidden_states: [B, L, HC_MULT, D] - 当前 Transformer 层的隐状态（作为 Query）
        input_ids: [B, L] - 输入 token IDs
        """
        
        # ---------------------------------------------------------------------
        # Step 1: 检索 (Retrieval)
        # ---------------------------------------------------------------------
        # 计算哈希索引 (CPU/Numpy 操作，实际中会在 GPU Kernel 完成或 CPU 预取)
        hash_indices = self.hash_mapping.hash(input_ids)[self.layer_id]
        hash_input_ids = torch.from_numpy(hash_indices).to(hidden_states.device) # [B, L, Num_Heads]
        
        # 查表得到 N-gram Embeddings
        # Flatten: 将所有 N-gram 类型和 Heads 展平，准备拼接
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2) 
        # embeddings 形状: [B, L, Engram_Hidden_Size]
        
        # ---------------------------------------------------------------------
        # Step 2: 上下文感知门控 (Context-aware Gating)
        # ---------------------------------------------------------------------
        gates = []
        for hc_idx in range(backbone_config.hc_mult):
            # Key: 来自静态记忆 (Engram)
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            
            # Query: 来自当前动态上下文 (Hidden State)
            query = hidden_states[:,:,hc_idx,:]
            normed_query = self.norm2[hc_idx](query)
            
            # Dot Product Attention 风格的门控计算
            # 计算相似度: 如果当前上下文和检索到的静态知识匹配，则 gate 值高
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(backbone_config.hidden_size)
            
            # 论文公式 (4) 的实现: RMSNorm + Sigmoid
            # 这里的 abs().sqrt() 是为了调整数值范围，使其更平滑
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1) # [B, L, 1]
            gates.append(gate)
        
        # 堆叠所有分支的门控: [B, L, HC_MULT, 1]
        gates = torch.stack(gates, dim=2)
        
        # ---------------------------------------------------------------------
        # Step 3: 融合 (Fusion)
        # ---------------------------------------------------------------------
        # Value: 共享的 Value 投影
        value_proj = self.value_proj(embeddings).unsqueeze(2) # [B, L, 1, D]
        
        # 加权: Gate * Value
        # 这一步决定了“多少静态记忆”被注入到模型中
        gated_value = gates * value_proj 
        
        # 加上短卷积层，并在最后通过残差连接输出
        output = gated_value + self.short_conv(gated_value)
        
        return output 

# =============================================================================
# 7. 模拟 Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        # 模拟 Attention 和 MoE，直接透传 (Identity)
        self.attn = lambda x:x
        self.moe  = lambda x:x
        
        self.engram = None
        # 仅在配置指定的层（如 layer 1, 15）插入 Engram
        if layer_id in engram_cfg.layer_ids:
            self.engram = Engram(layer_id=layer_id)
    
    def forward(self, input_ids, hidden_states):
        # 如果当前层有 Engram，则执行：Hidden = Hidden + Engram(Hidden, Inputs)
        if self.engram is not None:
            engram_out = self.engram(hidden_states=hidden_states, input_ids=input_ids)
            hidden_states = engram_out + hidden_states
            
        # 标准 Transformer 流程
        hidden_states = self.attn(hidden_states) + hidden_states
        hidden_states = self.moe(hidden_states) + hidden_states
        return hidden_states

# =============================================================================
# 8. 主程序入口
# =============================================================================

if __name__ == '__main__':
    # 构建简易 LLM
    LLM = [
        nn.Embedding(backbone_config.vocab_size, backbone_config.hidden_size),
        *[TransformerBlock(layer_id=layer_id) for layer_id in range(backbone_config.num_layers)],
        nn.Linear(backbone_config.hidden_size, backbone_config.vocab_size)
    ]

    text = "Only Alexander the Great could tame the horse Bucephalus."
    print(f"Input Text: {text}")
    
    # 真实的分词过程
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path, trust_remote_code=True)
    input_ids = tokenizer(text, return_tensors='pt').input_ids

    B, L = input_ids.shape
    print(f"Token IDs Shape: {input_ids.shape}")

    # 模拟前向传播循环
    hidden_states = None
    output = None
    
    for idx, layer in enumerate(LLM):
        if idx == 0:
            # Embedding 层
            hidden_states = LLM[0](input_ids)
            # 模拟 mHC: 将单一隐状态扩展为 [B, L, HC_MULT, D]
            hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, backbone_config.hc_mult, -1)      
        elif idx == len(LLM)-1:
            # LM Head 层
            # 模拟 mHC 归约: 简单取第一个分支
            hidden_states = hidden_states[:,:,0,:] 
            output = layer(hidden_states)
        else:
            # Transformer Block (含 Engram)
            hidden_states = layer(input_ids=input_ids, hidden_states=hidden_states)

    print("\n✅ Forward Complete! (前向传播完成)")
    print(f"Input Shape: {input_ids.shape}")
    print(f"Logits Shape: {output.shape}")