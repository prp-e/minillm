# Estimate total parameters dynamically
d_model = 384
n_heads = 8
n_layers = 6
d_ff = 1536
vocab = 5000

d_k = d_model // n_heads
attn_params = n_layers * (4 * d_model * d_model)        # q,k,v,o
ffn_params = n_layers * (2 * d_model * d_ff)            # up + down
emb_params = vocab * d_model                            # token embeddings
total_params = attn_params + ffn_params + emb_params
print(f"≈ {total_params/1e6:.1f}M parameters")