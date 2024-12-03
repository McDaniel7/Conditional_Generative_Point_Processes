# Conditional Generative Modeling for High-dimensional Marked Temporal Point Processes

Point processes offer a versatile framework for sequential event modeling. However, the computational challenges and constrained representational power of the existing point process models have impeded their potential for wider applications. This limitation becomes especially pronounced when dealing with event data that is associated with multi-dimensional or high-dimensional marks, such as texts or images. To address this challenge, this study proposes a novel event-generation framework for modeling point processes with high-dimensional marks. We aim to capture the distribution of events without explicitly specifying the conditional intensity or probability density function. Instead, we use a conditional generator that takes the history of events as input and generates the high-quality subsequent event that is likely to occur given the prior observations. The proposed framework offers a host of benefits, including considerable representational power to capture intricate dynamics in multi- or even high-dimensional event space, as well as exceptional efficiency in learning the model and generating samples. Our numerical results demonstrate superior performance compared to other state-of-the-art baselines.

## Model

![](https://github.com/McDaniel7/Generative_PP/blob/main/results/model_illustration.png)

## Results

![](https://github.com/McDaniel7/Generative_PP/blob/main/results/real_data_generation.png)

## Usage

- `generative_pp_KDE/KDE_CEG.py` defines the generative point process learned through non-parametric learning.
- `generative_pp_VAE/VAE_CEG.py` defines the generative point process learned through variational learning.
- `generative_pp_CDDM/CDDM_CEG.py` defines the generative point process learned through denoising diffusion.
- `sampling.py` includes the efficient sequential event generation using our generative point process.

### Examples

Examples of generating sequential events in multi-dimensional space using `sampling.py`.

1. Generator learned by non-parametric learning:
```python
from KDE_CEG import NeuralPP
from sampling import KDE_NPP_data_generator

# test_config can be defined accordingly in different experiments
test_config = {
    'data': 'ETAS',
    'data_path': data_path,
    'train_size': train_size,
    'data_dim': 3,
    'hid_dim': 32,
    'mlp_layer': 2,
    'kde_bdw':[5., 5., 5.],
    'kde_bdw_base':[0.2, 0.05, 0.05],
    'kde_bdw_decay':0.9,
    'n_samples': 1000,
    'noise_dim': 10,
    'mlp_dim': 32,
    'batch_size':1,
    'epochs': 100,
    'lr': 1e-3,
    'dropout': 0.1,
    'prt_evry':1,
    'early_stop': False,
    'alpha': 0.05,
    'log_mode': False,
    'log_iter': 5,
    'saved_path': saved_path,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

model = NeuralPP(test_config)
model.load_state_dict(torch.load(test_config["saved_path"]))

batch_size = 1000
seq_len = 150

seed = 100
torch.manual_seed(seed)
data = KDE_NPP_data_generator(model, batch_size, seq_len)
```

2. Generator learned by variational learning:
```python
from VAE_CEG import NeuralPP
from sampling import VAE_NPP_data_generator

# test_config can be defined accordingly in different experiments
test_config = {
    'data': 'NCEDC',
    'data_path': data_path,
    'data_dim': 3,
    'hid_dim': 32,
    'noise_dim': 256,
    'mlp_layer': 1,
    'mlp_dim': 3000,
    'batch_size':1,
    'epochs': 50,
    'lr': 1e-3,
    'prt_evry':1,
    'early_stop': False,
    'alpha':0.05,
    'log_mode':False,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'saved_path': saved_path,
}

model = NeuralPP(test_config)
model.load_state_dict(torch.load(test_config["saved_path"]))

batch_size = 1000
seq_len = 150

seed = 100
torch.manual_seed(seed)
data = VAE_NPP_data_generator(model, batch_size, seq_len)
```


## Citation

```
@article{dong2023conditional,
  title={Conditional Generative Modeling is All You Need for Marked Temporal Point Processes},
  author={Dong, Zheng and Fan, Zekai and Zhu, Shixiang},
  journal={arXiv preprint arXiv:2305.12569},
  year={2023}
}
```
