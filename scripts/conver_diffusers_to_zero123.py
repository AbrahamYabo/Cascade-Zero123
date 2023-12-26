import argparse
import torch
import sys
sys.path.insert(0, '../')
from pipeline_zero1to3 import CCProjection, Zero1to3StableDiffusionPipeline
import os.path as osp

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

from diffusers.models import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import logging

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

logger = logging.get_logger(__name__)


def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming to them. It splits
    attention layers, and takes into account additional replacements that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace("mid_block.resnets.0","middle_block.0")
        new_path = new_path.replace("mid_block.attentions.0","middle_block.1")
        new_path = new_path.replace("mid_block.resnets.1","middle_block.2")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        checkpoint[new_path] = old_checkpoint[path["old"]]

def vae_assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming to them. It splits
    attention layers, and takes into account additional replacements that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        print("**attention_paths_to_split " + str(attention_paths_to_split))

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace("mid_block.resnets.0","mid.block_1")
        new_path = new_path.replace("mid_block.resnets.1","mid.block_2")
        new_path = new_path.replace("mid_block.attentions.0","mid.attn_1")
        ##
        new_path = new_path.replace("up_blocks.0.resnets.0","up.3.block.0")
        new_path = new_path.replace("up_blocks.0.resnets.1","up.3.block.1")
        new_path = new_path.replace("up_blocks.0.resnets.2","up.3.block.2")

        new_path = new_path.replace("up_blocks.1.resnets.0","up.2.block.0")
        new_path = new_path.replace("up_blocks.1.resnets.1","up.2.block.1")
        new_path = new_path.replace("up_blocks.1.resnets.2","up.2.block.2")

        new_path = new_path.replace("up_blocks.2.resnets.0","up.1.block.0")
        new_path = new_path.replace("up_blocks.2.resnets.1","up.1.block.1")
        new_path = new_path.replace("up_blocks.2.resnets.2","up.1.block.2")

        new_path = new_path.replace("up_blocks.3.resnets.0","up.0.block.0")
        new_path = new_path.replace("up_blocks.3.resnets.1","up.0.block.1")
        new_path = new_path.replace("up_blocks.3.resnets.2","up.0.block.2")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        checkpoint[new_path] = old_checkpoint[path["old"]]


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("norm1", "in_layers.0")
        new_item = new_item.replace("conv1", "in_layers.2")

        new_item = new_item.replace("norm2", "out_layers.0")
        new_item = new_item.replace("conv2", "out_layers.3")

        new_item = new_item.replace("time_emb_proj", "emb_layers.1")
        new_item = new_item.replace("conv_shortcut", "skip_connection")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping

def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        mapping.append({"old": old_item, "new": new_item})

    return mapping

def convert_unet_state_dict(
    checkpoint,template_unet_state_dict, config, path=None, extract_ema=False, controlnet=False, skip_extract_state_dict=False
):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """

    if skip_extract_state_dict:
        unet_state_dict = checkpoint
    
    new_checkpoint = {}

    new_checkpoint["time_embed.0.weight"] = unet_state_dict["time_embedding.linear_1.weight"]
    new_checkpoint["time_embed.0.bias"] = unet_state_dict["time_embedding.linear_1.bias"]
    new_checkpoint["time_embed.2.weight"] = unet_state_dict["time_embedding.linear_2.weight"]
    new_checkpoint["time_embed.2.bias"] = unet_state_dict["time_embedding.linear_2.bias"]

    if config["class_embed_type"] is None:
        # No parameters to port
        ...
    elif config["class_embed_type"] == "timestep" or config["class_embed_type"] == "projection":
        new_checkpoint["label_emb.0.0.weight"] = unet_state_dict["class_embedding.linear_1.weight"]
        new_checkpoint["label_emb.0.0.bias"] = unet_state_dict["class_embedding.linear_1.bias"]
        new_checkpoint["label_emb.0.2.weight"] = unet_state_dict["class_embedding.linear_2.weight"]
        new_checkpoint["label_emb.0.2.bias"] = unet_state_dict["class_embedding.linear_2.bias"]
    else:
        raise NotImplementedError(f"Not implemented `class_embed_type`: {config['class_embed_type']}")

    if config["addition_embed_type"] == "text_time":
        new_checkpoint["label_emb.0.0.weight"] = unet_state_dict["add_embedding.linear_1.weight"]
        new_checkpoint["label_emb.0.0.bias"] = unet_state_dict["add_embedding.linear_1.bias"]
        new_checkpoint["label_emb.0.2.weight"] = unet_state_dict["add_embedding.linear_2.weight"]
        new_checkpoint["label_emb.0.2.bias"] = unet_state_dict["add_embedding.linear_2.bias"]

    new_checkpoint["input_blocks.0.0.weight"] = unet_state_dict["conv_in.weight"]
    new_checkpoint["input_blocks.0.0.bias"] = unet_state_dict["conv_in.bias"]

    if not controlnet:
        new_checkpoint["out.0.weight"] = unet_state_dict["conv_norm_out.weight"]
        new_checkpoint["out.0.bias"] = unet_state_dict["conv_norm_out.bias"]
        new_checkpoint["out.2.weight"] = unet_state_dict["conv_out.weight"]
        new_checkpoint["out.2.bias"] = unet_state_dict["conv_out.bias"]

    ###########################################################
    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "down_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"down_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }
    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "mid_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"mid_block.resnets.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    middle_blocks_attn = [
        key for key in unet_state_dict if f"mid_block.attentions." in key
    ]

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "up_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"up_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    for i in range(1, 12):
        block_id = (i - 1) // (config["layers_per_block"] + 1)

        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[block_id] if f"down_blocks.{block_id}.resnets" in key
        ]
        attentions = [key for key in input_blocks[block_id] if f"down_blocks.{block_id}.attentions" in key]

        if f"down_blocks.{block_id}.downsamplers.0.conv.weight" in unet_state_dict and layer_in_block_id == 2:
            new_checkpoint[f"input_blocks.{i}.0.op.weight"] = unet_state_dict.pop(
                f"down_blocks.{block_id}.downsamplers.0.conv.weight"
            )
            new_checkpoint[f"input_blocks.{i}.0.op.bias"] = unet_state_dict.pop(
                f"down_blocks.{block_id}.downsamplers.0.conv.bias"
            )

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"down_blocks.{block_id}.resnets.{layer_in_block_id}", "new": f"input_blocks.{i}.0"}
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {"old": f"down_blocks.{block_id}.attentions.{layer_in_block_id}", "new": f"input_blocks.{i}.1"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks_attn
    resnet_1 = middle_blocks[1]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "mid_block.attentions.0", "new": "middle_block.1"}
    assign_to_checkpoint(
        attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )

    for i in range(0, 12):#start at 0
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[block_id]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[block_id] if f"up_blocks.{block_id}.resnets.{layer_in_block_id}" in key]
            attentions = [key for key in output_blocks[block_id] if f"up_blocks.{block_id}.attentions.{layer_in_block_id}" in key]
            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {"old": f"up_blocks.{block_id}.resnets.{layer_in_block_id}", "new": f"output_blocks.{i}.0"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

            output_block_list = {k: sorted(v) for k, v in output_block_list.items()}
            if ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.bias", "conv.weight"])
                new_checkpoint[f"output_blocks.{i}.{index}.conv.weight"] = unet_state_dict[
                f"up_blocks.{block_id}.upsamplers.0.conv.weight"
                ]
                new_checkpoint[f"output_blocks.{i}.{index}.conv.bias"] = unet_state_dict[
                f"up_blocks.{block_id}.upsamplers.0.conv.bias"
                ]

                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                    "new": f"output_blocks.{i}.1",
                }
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )
            
            if i==2:
                paths = [{'old': 'up_blocks.0.upsamplers.0.conv.weight', 'new': 'up_blocks.0.upsamplers.0.conv.weight'},{'old': 'up_blocks.0.upsamplers.0.conv.bias', 'new': 'up_blocks.0.upsamplers.0.conv.bias'}]
                meta_path = {
                    "old": f"up_blocks.0.upsamplers.0",
                    "new": f"output_blocks.2.1",
                }
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )
            if i==5:
                paths = [{'old': 'up_blocks.1.upsamplers.0.conv.weight', 'new': 'up_blocks.1.upsamplers.0.conv.weight'},{'old': 'up_blocks.1.upsamplers.0.conv.bias', 'new': 'up_blocks.1.upsamplers.0.conv.bias'}]
                meta_path = {
                    "old": f"up_blocks.1.upsamplers.0",
                    "new": f"output_blocks.5.2",
                }
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )                
            if i==8:
                paths = [{'old': 'up_blocks.2.upsamplers.0.conv.weight', 'new': 'up_blocks.2.upsamplers.0.conv.weight'},{'old': 'up_blocks.2.upsamplers.0.conv.bias', 'new': 'up_blocks.2.upsamplers.0.conv.bias'}]
                meta_path = {
                    "old": f"up_blocks.2.upsamplers.0",
                    "new": f"output_blocks.8.2",
                }
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )                
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["old"]])
                new_path = ".".join(["output_blocks", str(i), path["new"]])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    if controlnet:
        # conditioning embedding

        orig_index = 0

        new_checkpoint[f"input_hint_block.{orig_index}.weight"] = unet_state_dict.pop(
            "controlnet_cond_embedding.conv_in.weight"
        )
        new_checkpoint[f"input_hint_block.{orig_index}.bias"] = unet_state_dict.pop(
            "controlnet_cond_embedding.conv_in.bias"
        )

        orig_index += 2

        diffusers_index = 0

        while diffusers_index < 6:
            new_checkpoint[f"input_hint_block.{orig_index}.weight"] = unet_state_dict.pop(
                f"controlnet_cond_embedding.blocks.{diffusers_index}.weight"
            )
            new_checkpoint[f"input_hint_block.{orig_index}.bias"] = unet_state_dict.pop(
                f"controlnet_cond_embedding.blocks.{diffusers_index}.bias"
            )
            diffusers_index += 1
            orig_index += 2

        new_checkpoint[f"input_hint_block.{orig_index}.weight"] = unet_state_dict.pop(
            "controlnet_cond_embedding.conv_out.weight"
        )
        new_checkpoint[f"input_hint_block.{orig_index}.bias"] = unet_state_dict.pop(
            "controlnet_cond_embedding.conv_out.bias"
        )

        # down blocks
        for i in range(num_input_blocks):
            new_checkpoint[f"zero_convs.{i}.0.weight"] = unet_state_dict.pop(f"controlnet_down_blocks.{i}.weight")
            new_checkpoint[f"zero_convs.{i}.0.bias"] = unet_state_dict.pop(f"controlnet_down_blocks.{i}.bias")

        # mid block
        new_checkpoint["middle_block_out.0.weight"] = unet_state_dict.pop("controlnet_mid_block.weight")
        new_checkpoint["middle_block_out.0.bias"] = unet_state_dict.pop("controlnet_mid_block.bias")

    return new_checkpoint



def convert_vae_state_dict(checkpoint, config):
    vae_state_dict = checkpoint

    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.norm_out.weight"] = vae_state_dict["encoder.conv_norm_out.weight"]
    new_checkpoint["encoder.norm_out.bias"] = vae_state_dict["encoder.conv_norm_out.bias"]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.norm_out.weight"] = vae_state_dict["decoder.conv_norm_out.weight"]
    new_checkpoint["decoder.norm_out.bias"] = vae_state_dict["decoder.conv_norm_out.bias"]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down_blocks" in layer})
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down_blocks.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    ##################################################
    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up_blocks" in layer})
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up_blocks.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down_blocks.{i}" in key and f"down_blocks.{i}.downsamplers" not in key]
        if f"encoder.down_blocks.{i}.downsamplers.0.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down.{i}.downsample.conv.weight"] = vae_state_dict.pop(
                f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"
            )
            new_checkpoint[f"encoder.down.{i}.downsample.conv.bias"] = vae_state_dict.pop(
                f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down_blocks.{i}.resnets", "new": f"down.{i}.block"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "encoder.mid_block.resnets" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"mid_block.resnets.{i - 1}" in key]
        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid_block.resnets.{i - 1}", "new": f"mid.block_{i}"}
        vae_assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "encoder.mid_block.attentions" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid_block.attentions.0", "new": "mid.attn_1"}
    vae_assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear_reverse(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up_blocks.{block_id}" in key and f"up_blocks.{block_id}.upsamplers" not in key
        ]

        if f"decoder.up_blocks.{i}.upsamplers.0.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up.{block_id}.upsample.conv.weight"] = vae_state_dict[
                f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"
            ]
            new_checkpoint[f"decoder.up.{block_id}.upsample.conv.bias"] = vae_state_dict[
                f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"
            ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up_blocks.{i}.resnets", "new": f"up.{block_id}.block"}
        vae_assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "decoder.mid_block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid_block.resnets.{i - 1}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid_block.resnets.{i - 1}", "new": f"mid.block_{i}"}
        vae_assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "decoder.mid_block.attentions" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid_block.attentions.0", "new": "mid.attn_1"}
    vae_assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear_reverse(new_checkpoint)
    return new_checkpoint


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("conv_shortcut", "nin_shortcut")    
        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping

def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("group_norm.weight", "norm.weight")
        new_item = new_item.replace("group_norm.bias", "norm.bias")

        new_item = new_item.replace("to_q.weight", "q.weight")
        new_item = new_item.replace("to_q.bias", "q.bias")

        new_item = new_item.replace("to_k.weight", "k.weight")
        new_item = new_item.replace("to_k.bias", "k.bias")

        new_item = new_item.replace("to_v.weight", "v.weight")
        new_item = new_item.replace("to_v.bias", "v.bias")

        new_item = new_item.replace("to_out.0.weight", "proj_out.weight")
        new_item = new_item.replace("to_out.0.bias", "proj_out.bias")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        ##################################################
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]

def conv_attn_to_linear_reverse(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["q.weight", "k.weight", "v.weight"]
    for key in keys:
        ##################################################
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim == 2:
                checkpoint[key] = checkpoint[key].unsqueeze(-1).unsqueeze(-1)
        elif 'proj_out.weight' in key:
            if checkpoint[key].ndim == 2:
                checkpoint[key] = checkpoint[key].unsqueeze(-1).unsqueeze(-1)

def get_template_zero123_unet_dict(template_checkpoint):
    template_unet_state_dict = {}
    keys = list(template_checkpoint.keys())
    unet_key = "model.diffusion_model."

    for key in keys:
        if key.startswith(unet_key):
            template_unet_state_dict[key.replace(unet_key, "")] = template_checkpoint[key]
    return template_unet_state_dict

def get_template_zero123_vae_dict(template_checkpoint):
    vae_state_dict = {}
    vae_key = "first_stage_model."
    keys = list(template_checkpoint.keys())
    for key in keys:
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = template_checkpoint.get(key)
    return vae_state_dict


def convert_from_diffusers_ckpt(model_path,template_zero123_checkpoint_path,device):
    diffusers_path=model_path
    template_zero123_ckpt=torch.load(template_zero123_checkpoint_path, map_location=device)
    template_checkpoint=template_zero123_ckpt['state_dict']
    ##1. unet
    unet_path = osp.join(model_path, "unet", "diffusion_pytorch_model.bin")
    unet_state_dict = torch.load(unet_path, map_location="cpu")
    
    ##diffusers unet
    temp_unet_state_dict=unet_state_dict

    unet_config = OmegaConf.load(osp.join(args.model_path, "unet", "config.json"))    
    #zero_unet template
    template_zero123_unet_dict=get_template_zero123_unet_dict(template_checkpoint)
    
    ##
    unet_state_dict = convert_unet_state_dict(unet_state_dict,template_zero123_unet_dict,unet_config, skip_extract_state_dict=True)
    
    ##
    for k in temp_unet_state_dict.keys():##
        if k not in unet_state_dict.keys() and k in template_zero123_unet_dict.keys():
            print('***diffusers unet not included',k)    

    unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}
    
    # check
    misincluded_unet_key=[]
    for k in unet_state_dict.keys():
        if k not in template_checkpoint.keys():
            misincluded_unet_key.append(k)
    for e in misincluded_unet_key:
        unet_state_dict.pop(e)
    # check
    for k in template_checkpoint.keys():
        if k.startswith("model.diffusion_model"):
            if k not in unet_state_dict.keys():
                print('***unet not included:',k)

    #2. vae
    vae_path = osp.join(args.model_path, "vae", "diffusion_pytorch_model.bin")
    vae_state_dict = torch.load(vae_path, map_location="cpu")
    vae_config = OmegaConf.load(osp.join(args.model_path, "vae", "config.json"))

    #temp vae state dict
    temp_vae_state_dict=vae_state_dict

    template_zero123_vae_dict=get_template_zero123_vae_dict(template_checkpoint)
    ##complete
    vae_state_dict = convert_vae_state_dict(vae_state_dict, vae_config)

    ##check
    for k in temp_vae_state_dict.keys():
        if k not in vae_state_dict.keys() and k in template_zero123_vae_dict.keys():
           print('***diffusers vae not included:',k)

    vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}
    
    ##check
    misincluded_vae_key=[]
    for k in vae_state_dict.keys():
        if k not in template_checkpoint.keys():
            misincluded_vae_key.append(k)
    for e in misincluded_vae_key:
        vae_state_dict.pop(e)
    
    #check
    for k in template_checkpoint.keys():
        if k.startswith("first_stage_model."):
            if k not in vae_state_dict.keys():
                print('***vae not included:',k)


    # ##3.clip and ccprojection
    cc_proj_path = osp.join(args.model_path, "cc_projection", "diffusion_pytorch_model.bin")
    cc_proj_dict = torch.load(cc_proj_path, map_location="cpu")
    cc_projection = {}
    cc_projection['cc_projection.weight'] = cc_proj_dict['projection.weight']
    cc_projection['cc_projection.bias'] = cc_proj_dict['projection.bias']

    state_dict = {**unet_state_dict, **vae_state_dict, **cc_projection}
    ##check
    for k in state_dict.keys():
        if k not in template_checkpoint.keys():
            print('***keys mis-included',k)
    #check
    for k in template_checkpoint.keys():
        if k not in state_dict.keys():
            state_dict[k]=template_checkpoint[k]

    return state_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--template_checkpoint_path", default=None, type=str, required=True, help="Path to the template zero123 checkpoint."
    )
    parser.add_argument(
        "--original_config_file",
        default=None,
        type=str,
        help="The YAML config file corresponding to the original architecture.",
    )

    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")    
    parser.add_argument("--device", type=str, help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")

    parser.add_argument("--model_path", default=None, type=str, required=True, help="Path to the model to convert.")
    parser.add_argument(
        "--use_safetensors", action="store_true", help="Save weights use safetensors, default is ckpt."
    )

    args = parser.parse_args()


    assert args.model_path is not None, "Must provide a model path!"
    assert args.checkpoint_path is not None, "Must provide a checkpoint path!"
    
    state_dict=convert_from_diffusers_ckpt(model_path=args.model_path,template_zero123_checkpoint_path=args.template_checkpoint_path,device=args.device)
    if args.half:
        state_dict = {k: v.half() for k, v in state_dict.items()}

    state_dict = {"state_dict": state_dict}
    torch.save(state_dict, args.checkpoint_path)
