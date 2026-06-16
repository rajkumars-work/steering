class ExpertCrossAttention(nn.Module):
    """
    Cross-attention where the encoder KV projection is
    conditioned on the WP DOF class of the current token.
    Allows different DOF classes to 'look at' the encoder differently.
    """
    def __init__(self, edim, heads, n_experts=4, dropout=0.1):
        super().__init__()
        self.n_head    = heads
        self.q_proj    = nn.Linear(edim, edim)
        # One KV projection per DOF class
        self.kv_projs  = nn.ModuleList([
            nn.Linear(edim, 2 * edim) for _ in range(n_experts)
        ])
        self.wp_embed  = nn.Embedding(4, edim)
        self.gate      = nn.Linear(edim, n_experts)
        self.c_proj    = nn.Linear(edim, edim)
        self.dropout   = dropout

    def forward(self, x, enc_out, wp_class, key_padding_mask=None):
        B, T, C   = x.shape
        T_enc     = enc_out.size(1)
        q         = self.q_proj(x)

        # Soft-mix KV projections weighted by WP class
        wp_e      = self.wp_embed(wp_class)           # (B, T, C)
        kv_weights = F.softmax(
            self.gate(x + wp_e), dim=-1
        )                                              # (B, T, n_experts)

        # Weighted KV
        all_kv    = torch.stack(
            [proj(enc_out) for proj in self.kv_projs], dim=-1
        )                                              # (B, T_enc, 2C, n_experts)
        kv        = (all_kv * kv_weights[:, :1, None, :]).sum(-1)
        # Note: kv_weights averaged over T for enc projection; or see note below
        k, v      = kv.split(C, dim=-1)
        # ... standard scaled dot product attention ...
