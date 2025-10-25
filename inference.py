from transformers import AutoTokenizer
import math, torch
import torch.nn.functional as F
import random, numpy as np
import argparse

def set_seed(seed=42):
    """Fix random seeds to keep inference deterministic where possible."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def repeat_kv(hidden_states, n_rep):
    """Repeat key/value heads (view-based) to match query heads in GQA."""
    b, n_kv, s, d = hidden_states.shape
    if n_rep == 1: return hidden_states
    hs = hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return hs.reshape(b, n_kv * n_rep, s, d)

def rotary_emb(dim, max_seq_len):
    """Precompute RoPE cos/sin tables for dim and max_seq_len."""
    ang = (1/10000)**torch.linspace(0,1,steps=dim//4, dtype=torch.float32)
    ang = torch.cat([ang, torch.zeros_like(ang)])
    t = torch.arange(max_seq_len, dtype=torch.float32)
    theta = torch.einsum("i,j->ij", t, ang)
    return theta.cos(), theta.sin()

def apply_rotary(x, cos, sin):
    """Apply RoPE rotation to tensor shaped (B,T,H,D)."""
    x1, x2 = x.chunk(2, dim=-1)
    T = x.size(1)
    cos, sin = cos[:T], sin[:T]
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)

def qwen_attention(x, attn, cos, sin, dropout=0.0):
    """Qwen-style attention with QK-norm, RoPE, GQA, Flash-attn."""
    b,t,d = x.shape
    n_heads, n_kv, d_k, d_model = attn["n_heads"], attn["n_kv"], attn["d_k"], attn["d_model"]
    q = F.linear(x, attn["wq"])  
    k = F.linear(x, attn["wk"])  
    v = F.linear(x, attn["wv"])  
    q = q.view(b,t,n_heads,d_k); k = k.view(b,t,n_kv,d_k); v = v.view(b,t,n_kv,d_k)
    q = F.normalize(q, dim=-1);  k = F.normalize(k, dim=-1)
    q = apply_rotary(q, cos, sin); k = apply_rotary(k, cos, sin)
    Q, K, V = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    K = repeat_kv(K, attn["n_groups"]); V = repeat_kv(V, attn["n_groups"])
    out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=dropout)
    out = out.transpose(1,2).contiguous().view(b,t,d_model)
    return F.linear(out, attn["wo"])

def swiglu_ffn(x, w_up, w_gate, w_down, dropout=0.0):
    """Feed-forward with SwiGLU activation."""
    a = F.silu(F.linear(x, w_gate)) * F.linear(x, w_up)
    return F.linear(F.dropout(a, dropout, training=False), w_down)

def transformer_block(x, weights, cos, sin, cfg):
    """One transformer block: LN -> Attn -> residual, LN -> FFN -> residual."""
    x_norm = F.layer_norm(x, [cfg["d_model"]])
    attn_out = qwen_attention(x_norm, weights["attn"], cos, sin, dropout=0.0)
    x = x + attn_out
    ff_norm = F.layer_norm(x, [cfg["d_model"]])
    ff_out = swiglu_ffn(ff_norm, **weights["ffn"], dropout=0.0)
    return x + ff_out

def forward_model(input_ids, tok_emb, blocks, lm_head, cos, sin, cfg):
    """Forward pass from token ids to logits using functional weights."""
    x = F.embedding(input_ids, tok_emb) * math.sqrt(cfg["d_model"])
    for b in blocks:
        x = transformer_block(x, b, cos, sin, cfg)
    x = F.layer_norm(x, [cfg["d_model"]])
    return F.linear(x, lm_head)

def load_trained_state(model_path, device, max_seq_len=512):
    """Load weights saved by train.py and build a minimal cfg from shapes."""
    ckpt = torch.load(model_path, map_location="cpu")
    tok_emb = ckpt["tok_emb"].to(device)
    lm_head = tok_emb  # weight tying
    blocks = ckpt["blocks"]
    for b in blocks:
        for k in ("wq","wk","wv","wo"):
            b["attn"][k] = b["attn"][k].to(device)
        for k in ("w_up","w_gate","w_down"):
            b["ffn"][k] = b["ffn"][k].to(device)

    d_model = blocks[0]["attn"]["d_model"]
    n_heads = blocks[0]["attn"]["n_heads"]
    n_kv    = blocks[0]["attn"]["n_kv"]
    d_k     = blocks[0]["attn"]["d_k"]
    n_layers= len(blocks)
    d_ff    = blocks[0]["ffn"]["w_up"].shape[0]
    vocab   = tok_emb.shape[0]

    cfg = {
        "d_model": d_model, "n_heads": n_heads, "n_kv": n_kv,
        "d_k": d_k, "n_layers": n_layers, "d_ff": d_ff,
        "vocab": vocab, "seq_len": max_seq_len
    }
    cos, sin = rotary_emb(d_model//n_heads, max_seq_len)
    cos, sin = cos.to(device), sin.to(device)
    return (tok_emb, blocks, lm_head, cos, sin), cfg

def generate_text(state, cfg, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.9, eos_token_id=None):
    """Autoregressive text generation (top-k/top-p/temperature)."""
    tok_emb, blocks, lm_head, cos, sin = state
    device = tok_emb.device
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    generated = input_ids.clone()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = forward_model(generated, tok_emb, blocks, lm_head, cos, sin, cfg)
            next_logits = logits[0, -1, :] / max(temperature, 1e-5)

            if top_k and top_k > 0:
                vk, ik = torch.topk(next_logits, top_k)
                mask = torch.full_like(next_logits, float("-inf"))
                mask[ik] = vk
                next_logits = mask

            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumsum = torch.cumsum(probs, dim=-1)
                cutoff = cumsum > top_p
                cutoff[1:] = cutoff[:-1].clone()
                cutoff[0] = False
                next_logits[sorted_idx[cutoff]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)  # [1,1]
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def main():
    """Example CLI entrypoint for quick inference."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="best_model.pt")
    p.add_argument("--tokenizer", type=str, default="HuggingFaceTB/SmolLM-135M")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_seq_len", type=int, default=512)
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state, cfg = load_trained_state(args.model_path, device, max_seq_len=args.max_seq_len)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    text = generate_text(
        state, cfg, tok, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        eos_token_id=tok.eos_token_id
    )
    print(text)