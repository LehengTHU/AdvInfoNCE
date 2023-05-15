# AdvInfoNCE


## Overview

Code of "Empowering Collaborative Filtering Generalization via Principled Adversarial Contrastive Loss"


## Run the Code

- We provide implementation for various baselines presented in the paper.

- To run the code, first run the following command to install tools used in evaluation:

```
python setup.py build_ext --inplace
```

### LightGCN backbone

For models with LightGCN as backbone, use models with in-batch negative sampling strategy. For example:

- AdvInfoNCE Training:

```python
python train_AdvDRO.py --train_norm --pred_norm --modeltype AdvDRO --model_version embed --adv_version pknm --dataset kuairec2 --n_layers 2 --neg_sample 128 --tau 2 --lr 3e-5 --batch_size 2048 --adv_lr 5e-5 --eta_epochs 12 --dsc kuairec_sota
```

- INFONCE Training:

```
python main.py --train_norm --pred_norm --modeltype  INFONCE --dataset kuairec2 --n_layers 2 --batch_size 2048 --lr 3e-5 --neg_sample 128 --tau 2  --dsc infonce
```

Details of hyperparameter settings for various baselines can be found in the paper.

### MF backbone

For models with MF as backbone, use models with random negative sampling strategy. For example:

- AdvInfoNCE Training:

```
python train_AdvDRO.py --train_norm --pred_norm --modeltype AdvDRO --model_version embed --adv_version pknm --dataset tencent_synthetic --n_layers 0 --lr 1e-3 --batch_size 2048 --neg_sample 128 --tau 0.09 --adv_lr 5e-5 --eta_epochs 8 --patience 20 --dsc tencent_sota
```

- InfoNCE Training:

```
python main.py --train_norm --pred_norm --modeltype INFONCE --dataset tencent_synthetic --n_layers 0 --tau 0.09 --neg_sample 128 --batch_size 2048 --lr 1e-3 --dsc infonce
```


## Requirements

- python == 3.7.10

- pytorch == 1.12.1+cu102

- tensorflow == 1.14





