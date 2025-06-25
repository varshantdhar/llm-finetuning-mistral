import torch

def get_svd_adapter(W: torch.Tensor, rank: int = 8):
    """
    Perform truncated SVD on the weight matrix W to extract top-rank low-rank factors

    Args:
        W (torch.Tensor): The weight matrix to perform SVD on [out_dim, in_dim]
        rank (int): Number of singular values to keep

    Returns:
        A (torch.nn.Parameter): The low-rank adapter A matrix [rank, in_dim]
        B (torch.nn.Parameter): The low-rank adapter B matrix [rank, out_dim] (transposed already)
    """
    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        A = torch.nn.Parameter(U[:, :rank].clone())
        B = torch.nn.Parameter(Vh[:rank, :]).clone().T)
    return A, B

def apply_pissa_to_model(model, rank: int = 8, target_modules: list[str] = None):
    """
    Injects PiSSA SVD-based weights into all LoRA modules in the transformer layers.

    Args:
        model (torch.nn.Module): A PEFT-wrapped model (LoRA applied via get_peft_model) 
        rank (int): Desired rank of the low-rank factors
        target_modules (list[str]): The list of modules to apply PISSA to (e.g., q_proj, k_proj, v_proj, o_proj))
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    layers = model.base_model.model.layers

    for i, layer in enumerate(layers):
        for module_name in target_modules:
            try:
                proj_module = getattr(layer.self_attn, module_name)
                W = proj_module.weight.data # [out_dim, in_dim]

                A, B = get_svd_adapter(W, rank)

                #Inject the low-rank factors into the module
                proj_module.lora_A.default.weight.data = A
                proj_module.lora_B.default.weight.data = B

                print(f"Applied PiSSA to {module_name} in layer {i} with rank-{rank} SVD initialization")
            
            except AttributeError:
                print(f"Warning: {module_name} not found in layer {i}")
           