from transformers.pytorch_utils import Conv1D, prune_conv1d_layer, find_pruneable_heads_and_indices
import torch
from torch.nn.utils import prune


def model_heads_by_magnitude(model, percentage, num_heads_per_layer, layers):
    least_used_heads_per_layer = []

    layer_count = 0
    for name, module in model.named_modules():
        if isinstance(module, Conv1D) and 'attn' in name:
            if layer_count >= layers:
                break

            num_heads_to_prune = max(1, round(percentage * num_heads_per_layer))

            attention_weights = module.weight
            head_magnitudes = attention_weights.view(num_heads_per_layer, -1).norm(dim=1)

            sorted_indices = head_magnitudes.argsort()
            least_used_heads = sorted_indices[:num_heads_to_prune].tolist()
            least_used_heads_per_layer.append((layer_count, least_used_heads))

            layer_count += 1

    return least_used_heads_per_layer


def prune_heads(block, heads):
    if len(heads) == 0:
        return

    heads, index = find_pruneable_heads_and_indices(heads, block.num_heads, block.head_dim, block.pruned_heads)
    index_attn = torch.cat([index, index + block.split_size, index + (2 * block.split_size)])

    block.c_attn = prune_conv1d_layer(block.c_attn, index_attn, dim=1)
    block.c_proj = prune_conv1d_layer(block.c_proj, index, dim=0)

    block.split_size = (block.split_size // block.num_heads) * (block.num_heads - len(heads))
    block.num_heads -= len(heads)
    block.pruned_heads = block.pruned_heads.union(heads)


def prune_attention_heads(model, heads_to_prune):
    for name, module in model.named_modules():
        if isinstance(module, Conv1D):
            if 'transformer.h.' in name:
                block_index = int(name.split('.')[2])
                block = model.transformer.h[block_index]

                if 'attn' in name:
                    prune_heads(block.attn, heads_to_prune[block_index][1])

                # elif 'mlp' in name:
                #     # get the dimensions of the current tensor
                #     fc_weight_shape = block.mlp.c_fc.weight.shape
                #     proj_weight_shape = block.mlp.c_proj.weight.shape
                #
                #     print(f"Shape of c_fc weight: {fc_weight_shape}")
                #     print(f"Shape of c_proj weight: {proj_weight_shape}")
                #
                #     # ensure indices are valid for pruning
                #     fc_indices_dim1 = torch.LongTensor([i for i in range(0, fc_weight_shape[1], 2)])
                #     fc_indices_dim0 = torch.LongTensor([i for i in range(0, fc_weight_shape[0], 2)])
                #     proj_indices_dim1 = torch.LongTensor([i for i in range(0, proj_weight_shape[1], 2)])
                #     proj_indices_dim0 = torch.LongTensor([i for i in range(0, proj_weight_shape[0], 2)])
                #
                #     # check index validity before pruning
                #     if len(fc_indices_dim1) <= fc_weight_shape[1]:
                #         block.mlp.c_fc = prune_conv1d_layer(block.mlp.c_fc, fc_indices_dim1, dim=1)
                #     if len(fc_indices_dim0) <= fc_weight_shape[0]:
                #         block.mlp.c_fc = prune_conv1d_layer(block.mlp.c_fc, fc_indices_dim0, dim=0)
                #
                #     if len(proj_indices_dim1) <= proj_weight_shape[1]:
                #         block.mlp.c_proj = prune_conv1d_layer(block.mlp.c_proj, proj_indices_dim1, dim=1)
                #     if len(proj_indices_dim0) <= proj_weight_shape[0]:
                #         block.mlp.c_proj = prune_conv1d_layer(block.mlp.c_proj, proj_indices_dim0, dim=0)


def prune_model(model, prune_ratio, prune_heads=True, prune_fc=True):
    for name, module in model.named_modules():
        if prune_heads and isinstance(module, Conv1D) and "attn" in name:
            prune.ln_structured(module, name='weight', amount=prune_ratio, dim=0, n=float('-inf'))
            prune.remove(module, name='weight')

        if prune_fc and isinstance(module, Conv1D) and "mlp" in name:
            prune.ln_structured(module, name='weight', amount=prune_ratio, dim=0, n=float('-inf'))
            prune.remove(module, name='weight')