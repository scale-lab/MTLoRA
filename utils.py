# --------------------------------------------------------
# MTLoRA
# GitHub: https://github.com/scale-lab/MTLoRA
# Built upon Swin Transformer (https://github.com/microsoft/Swin-Transformer)
#
# Original file:
# Copyright (c) 2021 Microsoft
# Licensed under the MIT License
# Written by Ze Liu
#
# Modifications:
# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details)
# --------------------------------------------------------


import os
import torch
import torch.distributed as dist
from torch import inf
import errno

from PIL import Image
import numpy as np
import cv2
import imageio
import scipy.io as sio
import torch.nn.functional as F
from models.lora import map_old_state_dict_weights


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger, backbone=False, quiet=False):
    resume_path = config.MODEL.RESUME if not backbone else config.MODEL.RESUME_BACKBONE
    logger.info(
        f"==============> Resuming form {resume_path}....................")
    if resume_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            resume_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(resume_path, map_location='cpu')

    mtlora = config.MODEL.MTLORA
    mtlora_enabled = mtlora.ENABLED

    skip_decoder = config.TRAIN.SKIP_DECODER_CKPT

    model_state = {k: v for k, v in checkpoint["model"].items(
    ) if not k.startswith("decoders")} if skip_decoder else checkpoint["model"]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in model_state.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del model_state[k]

    if config.MODEL.UPDATE_RELATIVE_POSITION:
        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [
            k for k in model_state.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del model_state[k]

        # delete relative_coords_table since we always re-init it
        relative_position_index_keys = [
            k for k in model_state.keys() if "relative_coords_table" in k]
        for k in relative_position_index_keys:
            del model_state[k]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [
            k for k in model_state.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = model_state[k]
            relative_position_bias_table_current = model.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    # bicubic interpolate relative_position_bias_table if not match
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                        mode='bicubic')
                    model_state[k] = relative_position_bias_table_pretrained_resized.view(
                        nH2, L2).permute(1, 0)

        # bicubic interpolate absolute_pos_embed if not match
        absolute_pos_embed_keys = [
            k for k in model_state.keys() if "absolute_pos_embed" in k]
        for k in absolute_pos_embed_keys:
            # dpe
            absolute_pos_embed_pretrained = model_state[k]
            absolute_pos_embed_current = model.model_state()[k]
            _, L1, C1 = absolute_pos_embed_pretrained.size()
            _, L2, C2 = absolute_pos_embed_current.size()
            if C1 != C1:
                logger.warning(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(
                        -1, S1, S1, C1)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(
                        0, 3, 1, 2)
                    absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                        absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(
                        0, 2, 3, 1)
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(
                        1, 2)
                    model_state[k] = absolute_pos_embed_pretrained_resized

    if mtlora_enabled:
        mapping = {}
        trainable_layers = []
        if mtlora.QKV_ENABLED:
            trainable_layers.extend(["attn.qkv.weight", "attn.qkv.bias"])
        if mtlora.PROJ_ENABLED:
            trainable_layers.extend(["attn.proj.weight", "attn.proj.bias"])
        if mtlora.FC1_ENABLED:
            trainable_layers.extend(["mlp.fc1.weight", "mlp.fc1.bias"])
        if mtlora.FC2_ENABLED:
            trainable_layers.extend(["mlp.fc2.weight", "mlp.fc2.bias"])
        if mtlora.DOWNSAMPLER_ENABLED:
            trainable_layers.extend(["downsample.reduction.weight"])

        for k, v in model_state.items():
            last_three = ".".join(k.split(".")[-3:])
            prefix = ".".join(k.split(".")[:-3])
            if last_three in trainable_layers:
                weight_bias = last_three.split(".")[-1]
                layer_name = ".".join(last_three.split(".")[:-1])
                mapping[f"{prefix}.{layer_name}.{weight_bias}"] = f"{prefix}.{layer_name}.linear.{weight_bias}"
        if not len(mapping):
            print("No keys needs to be mapped for LoRA")
        model_state = map_old_state_dict_weights(
            model_state, mapping, "", config.MODEL.MTLORA.SPLIT_QKV)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if not quiet:
        if len(missing) > 0:
            logger.warning("=============Missing Keys==============")
            for k in missing:
                logger.warning(k)
        if len(unexpected) > 0:
            logger.warning("=============Unexpected Keys==============")
            for k in unexpected:
                logger.warning(k)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint and not skip_decoder:
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(
            f"=> loaded successfully '{resume_path}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_pretrained(config, model, logger):
    logger.info(
        f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [
        k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [
        k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(
                    nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [
        k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(
                    -1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(
                    0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(
                    0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(
                    1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(
                f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    save_name = f'ckpt_epoch_{epoch}.pth'
    save_path = os.path.join(config.OUTPUT, save_name)
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    return save_path


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d)
                                for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device)
                         for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                # unscale the gradients of optimizer's assigned params in-place
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def tens2image(tens, transpose=False):
    """Converts tensor with 2 or 3 dimensions to numpy array"""
    im = tens.cpu().detach().numpy()

    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)
    elif im.shape[-1] == 1:
        im = np.squeeze(im)
    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)
    if transpose:
        if im.ndim == 3:
            im = im.transpose((1, 2, 0))
    return im


def normalize(arr, t_min=0, t_max=255):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = arr.max() - arr.min()
    for i in arr:
        temp = (((i - arr.min())*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    res = np.array(norm_arr)
    return res


def save_imgs_mtl(batch_imgs, batch_labels, batch_predictions, path, id):
    import torchvision

    imgs = tens2image(batch_imgs, transpose=True)
    labels = {task: tens2image(label, transpose=True)
              for task, label in batch_labels.items()}
    predictions = {task: tens2image(prediction)
                   for task, prediction in batch_predictions.items()}

    Image.fromarray(normalize(imgs, 0, 255).astype(
        np.uint8)).save(f'{path}/{id}_img.png')

    for task in labels.keys():
        if task == "semseg":
            print(np.sum(labels[task] != 255))
            labels[task] = labels[task] != 255
            predictions[task] = predictions[task] != 225
            batch_imgs = 255*(batch_imgs-torch.min(batch_imgs)) / \
                (torch.max(batch_imgs)-torch.min(batch_imgs))
            semseg = torchvision.utils.draw_segmentation_masks(batch_imgs[0].cpu().detach().to(torch.uint8),
                                                               batch_predictions[task][0].to(torch.bool), colors="blue", alpha=0.5)
            Image.fromarray(semseg.numpy().transpose((1, 2, 0))
                            ).save(f'{path}/{id}_{task}_pred.png')
            semseg = torchvision.utils.draw_segmentation_masks(batch_imgs[0].cpu().detach().to(torch.uint8),
                                                               batch_labels[task][0].to(torch.bool), colors="blue", alpha=0.5)
            Image.fromarray(semseg.numpy().transpose((1, 2, 0))
                            ).save(f'{path}/{id}_{task}_gt.png')
        else:
            labels[task] = normalize(labels[task], 0, 255)
            predictions[task] = normalize(predictions[task], 0, 255)

            Image.fromarray(labels[task].astype(np.uint8)).save(
                f'{path}/{id}_{task}_gt.png')
            Image.fromarray(predictions[task].astype(np.uint8)).save(
                f'{path}/{id}_{task}_pred.png')
