from .build import build_loader as _build_loader, build_nyud, build_pascal
from .data_simmim_pt import build_loader_simmim
from .data_simmim_ft import build_loader_finetune


def build_loader(config, simmim=False, is_pretrain=False, val_only=False):
    if config.get('DATA', {}).get('NYUD', False):
        return build_nyud(config)
    if config.get('DATA', {}).get('PASCAL', False):
        return build_pascal(config, val_only)
    if not simmim:
        return _build_loader(config)
    if is_pretrain:
        return build_loader_simmim(config)
    else:
        return build_loader_finetune(config)
