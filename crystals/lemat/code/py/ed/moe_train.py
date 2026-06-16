for name, param in moe_model.named_parameters():
    param.requires_grad = "router" in name


for name, param in moe_model.named_parameters():
    param.requires_grad = (
        "mlp.experts" in name or
        "mlp.router"  in name
    )

for name, param in moe_model.named_parameters():
    param.requires_grad = True


def log_expert_utilisation(model: MoEEdGPT, x, wp_class, o=None):
    """
    Log per-expert weight statistics to confirm specialisation is occurring.
    Experts are specialising if: for a given wp_class,
    one expert consistently receives weight >> others.
    """
    model.eval()
    with torch.no_grad():
        # Hook into the router of the last decoder block
        last_block  = model.transformer.h[-1]
        router      = last_block.mlp.router
        # Run encoder
        enc_out = model.encode(o) if o is not None else None
        B, T    = x.shape
        pos     = torch.arange(T, device=x.device)
        x_emb   = model.transformer.wte(x) + model.transformer.wpe(pos)
        # Pass through all but last block
        for block in model.transformer.h[:-1]:
            x_emb = block(x_emb, enc_out, wp_class=wp_class)
        # Get routing weights from last block
        normed  = last_block.ln_2(x_emb)
        weights = router(normed, wp_class)   # (B, T, n_experts)

        for dof_class in range(4):
            mask = (wp_class == dof_class)    # (B, T) bool
            if mask.any():
                mean_w = weights[mask].mean(0) # (n_experts,)
                print(f"DOF class {dof_class}: expert weights = {mean_w.tolist()}")


# DOF class 0 (FIXED):   expert weights = [0.82, 0.06, 0.07, 0.05]
# DOF class 1 (LINE):    expert weights = [0.05, 0.79, 0.09, 0.07]
# DOF class 2 (PLANE):   expert weights = [0.06, 0.08, 0.78, 0.08]
# DOF class 3 (GENERAL): expert weights = [0.04, 0.07, 0.06, 0.83]

