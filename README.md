# STCL-BEV

**Spatio-Temporal Consistency Learning for BEV-Based Multi-View Multi-Target Detection and Tracking**

> **Notice**
> This repository contains the official implementation of the manuscript currently under submission to *The Visual Computer*.
> If you find this code useful, please consider citing our paper (see Citation section below).


## 🚀 Usage

### Getting Started
1. Install PyTorch with CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
2. Install mmcv with CUDA support

```bash
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

3. Build the deformable transformer (forked from Deformable DETR).

```bash
bash models/ops/make.sh
```

4. Install other dependencies.

```bash
pip install -r requirements.txt
```


### Training

```bash
python main.py fit -c configs/t_fit.yml \
    -c configs/d_{multiviewx,wildtrack}.yml
```

### Pretrained Weights

The pretrained model is available in the project release page.

Please download the checkpoint and place it under the `checkpoints/` directory.

### Evaluation

```bash
python main.py test -c checkpoints/{multiviewx,wildtrack}.yaml \
    --ckpt checkpoints/{multiviewx,wildtrack}.ckpt
```

---

## Acknowledgement

---
Simple-BEV: Adam W. Harley

MVDeTr: Yunzhong Hou

EarlyBird: Torben Teepe

---

## 📜 Citation

If you use this code, please cite our work:

```bibtex
@article{stcl2026,
  title={Spatio-Temporal Consistency Learning for BEV-Based Multi-View Multi-Target Detection and Tracking},
  author={Kangle Hu, Zhiqing Huang, Yanxin Zhang, Junpeng Zhang},
  journal={The Visual Computer},
  year={2026},
  publisher={Springer}
}
```

---

