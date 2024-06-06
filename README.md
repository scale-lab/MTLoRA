# MTLoRA: A Low-Rank Adaptation Approach for Efficient Multi-Task Learning

## Introduction

This is the official implementation of the paper: **MTLoRA: A Low-Rank Adaptation Approach for Efficient Multi-Task Learning** developed at [Brown University SCALE lab](https://scale-lab.github.io).

This repository provides a Python-based implementation of MTLoRA including [`MTLoRALinear`](models/lora.py) (the main module) and MTL architectures. 

The repository is built on top of [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and uses some modules from [Multi-Task-Learning-PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch).


## How to Run

Running MTLoRA code, is very simmilar to Swin's codebase:

1. **Clone the repository**
    ```bash
    git clone https://github.com/scale-lab/MTLoRA.git
    cd MTLoRA
    ```

2. **Install the prerequisites**
    - Install `PyTorch>=1.12.0` and `torchvision>=0.13.0` with `CUDA>=11.6`
    - Install dependencies: `pip install -r requirements.txt`

3. **Run the code**
        ```python
        python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --cfg configs/mtlora/tiny_448/<config>.yaml --pascal <path to pascal database> --tasks semseg,normals,sal,human_parts --batch-size <batch size> --ckpt-freq=20 --epoch=<num epochs> --resume-backbone <path to the weights of the chosen Swin variant>
        ```

  Swin variants and their weights can be found at the official [Swin Transformer repository](https://github.com/microsoft/Swin-Transformer). 
  
  The outputs will be saved in `output/` folder unless overridden by the argument `--output`.
  
## Authorship
Since the release commit is squashed, the GitHub contributors tab doesn't reflect the authors' contributions. The following authors contributed equally to this codebase:
- [Ahmed Agiza](https://github.com/ahmed-agiza)
- [Marina Neseem](https://github.com/marina-neseem)

## Citation
If you find MTLoRA helpful in your research, please cite our paper:
```
@inproceedings{agiza2024mtlora,
  title={MTLoRA: A Low-Rank Adaptation Approach for Efficient Multi-Task Learning},
  author={Agiza, Ahmed and Neseem, Marina and Reda, Sherief},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={},
  year={2024}
}
```

## License
MIT License. See [LICENSE](LICENSE) file
