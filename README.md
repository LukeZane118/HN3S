## HN3S: A Federated AutoEncoder Framework for Collaborative Filtering via Hybrid Negative Sampling and Secret Sharing

## 1. Overview
This repository is an PyTorch Implementation for "HN3S: A Federated AutoEncoder Framework for Collaborative Filtering via Hybrid Negative Sampling and Secret Sharing (under review)"

**Authors**: Lu Zhang, Guohui Li, Ling Yuan, Xuanang Ding, Qian Rong \
**Codes**: https://github.com/LukeZane118/HN3S

Note: this project is built upon [RecBole](https://github.com/RUCAIBox/RecBole), [rectorch](https://github.com/makgyver/rectorch) and [FMSS](https://github.com/LachlanLin/FMSS).

<a name="Environment"/>

## 2. Environment:

The code was developed and tested on the following python environment: 
```
python 3.8.13
pytorch 1.8.1
colorlog 6.6.0
colorama 0.4.5
pandas 1.2.3
numpy 1.21.5
scipy 1.9.0
munch 2.5.0
Bottleneck 1.3.4
scikit_learn 0.23.2
numba 0.55.2
```
<a name="instructions"/>

## 3. Instructions:

Train and evaluate HN3S:

- To evaluate HN3S on MultDAE
  - `bash ./run_multi_dae.sh`
- To evaluate HN3S on MultVAE
  - `bash ./run_multi_vae.sh`
- To evaluate HN3S on RecVAE
  - `bash ./run_recvae.sh`
- To evaluate HN3S on MacridVAE
  - `bash ./run_macridvae.sh`

<a name="citation"/>
