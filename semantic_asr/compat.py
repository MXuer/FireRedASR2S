def patch_torch_pytree_for_transformers() -> None:
    try:
        import torch.utils._pytree as pytree
    except ImportError:
        return

    if hasattr(pytree, "register_pytree_node"):
        return
    if not hasattr(pytree, "_register_pytree_node"):
        return

    original = pytree._register_pytree_node

    def register_pytree_node(node_type, flatten_fn, unflatten_fn, **kwargs):
        return original(node_type, flatten_fn, unflatten_fn)

    pytree.register_pytree_node = register_pytree_node

