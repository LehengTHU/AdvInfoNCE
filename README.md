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

- AdvInfoNCE Training:

Tencent:
```python
python train_AdvInfoNCE.py --train_norm --pred_norm --modeltype AdvInfoNCE --model_version embed --dataset tencent_synthetic --n_layers 2 --lr 1e-3 --batch_size 2048 --neg_sample 128 --tau 0.09 --adv_lr 5e-5 --eta_epochs 7 --k_neg 64 --patience 20 --dsc sota_tencent_lgn
```

KuaiRec:
```python
python train_AdvInfoNCE.py --train_norm --pred_norm --modeltype AdvInfoNCE --model_version embed  --dataset kuairec2 --n_layers 2 --neg_sample 128 --tau 2 --lr 3e-5 --batch_size 2048 --adv_lr 5e-5 --eta_epochs 12 --warm_up_epochs 0 --adv_interval 5 --dsc sota_kuairec_lgn
```

Yahoo:
```python
python train_AdvInfoNCE.py --train_norm --pred_norm --model_version embed --modeltype AdvInfoNCE --dataset yahoo.new --n_layers 2 --neg_sample 64 --tau 0.28 --lr 5e-4 --batch_size 1024 --adv_lr 1e-4 --eta_epochs 13 --dsc sota_yahoo_lgn
```

Coat:
```python
python train_AdvInfoNCE.py --train_norm --pred_norm --modeltype AdvInfoNCE --model_version embed --dataset coat --n_layers 2 --neg_sample 64 --tau 0.75 --lr 1e-3 --batch_size 1024 --adv_lr 1e-2 --eta_epochs 20 --adv_interval 15 --dsc sota_coat_lgn
```

- INFONCE Training:

```python
python main.py --train_norm --pred_norm --modeltype  INFONCE --dataset kuairec2 --n_layers 2 --batch_size 2048 --lr 3e-5 --neg_sample 128 --tau 2  --dsc infonce
```

### MF backbone

- AdvInfoNCE Training:

Tencent:
```python
python train_AdvInfoNCE.py --train_norm --pred_norm --modeltype AdvInfoNCE --model_version embed --dataset tencent_synthetic --n_layers 0 --lr 1e-3 --batch_size 2048 --neg_sample 128 --tau 0.09 --adv_lr 5e-5 --eta_epochs 8 --patience 20 --dsc sota_tencent
```

Yahoo:
```python
python train_AdvInfoNCE.py --train_norm --pred_norm --model_version embed --modeltype AdvInfoNCE --dataset yahoo.new --n_layers 0 --neg_sample 64 --tau 0.28 --lr 5e-4 --batch_size 1024 --adv_lr 1e-4 --eta_epochs 12 --dsc sota_yahoo
```

Coat:

```python
python train_AdvInfoNCE.py --train_norm --pred_norm --modeltype AdvInfoNCE --model_version embed --dataset coat --n_layers 0 --neg_sample 64 --tau 0.75 --lr 1e-3 --batch_size 1024 --adv_lr 1e-2 --eta_epochs 18 --adv_interval 15 --dsc sota_coat
```

- InfoNCE Training:

```python
python main.py --train_norm --pred_norm --modeltype INFONCE --dataset tencent_synthetic --n_layers 0 --tau 0.09 --neg_sample 128 --batch_size 2048 --lr 1e-3 --dsc infonce
```


## Requirements

- python == 3.7.10

- pytorch == 1.12.1+cu102

- tensorflow == 1.14





