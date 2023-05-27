# Conditional Generative Modeling is All You Need for Marked Temporal Point Processes

Recent advancements in generative modeling have made it possible to generate high-quality content from context information, but a key question remains: how to teach models to know when to generate content? To answer this question, this study proposes a novel event generative model that draws its statistical intuition from marked temporal point processes, and offers a clean, flexible, and computationally efficient solution for a wide range of applications involving multi-dimensional marks. We aim to capture the distribution of the point process without explicitly specifying the conditional intensity or probability density. Instead, we use a conditional generator that takes the history of events as input and generates the high-quality subsequent event that is likely to occur given the prior observations. The proposed framework offers a host of benefits, including exceptional efficiency in learning the model and generating samples, as well as considerable representational power to capture intricate dynamics in multi- or even high-dimensional event space. Our numerical results demonstrate superior performance compared to other state-of-the-art baselines.

## Model

![](https://github.com/McDaniel7/Generative_PP/blob/main/results/model_illustration.png)

## Results

![](https://github.com/McDaniel7/Generative_PP/blob/main/results/real_data_generation.png)

## Usage

- `generative_pp_KDE/KDE_CEG.py` defines the genrative point process learned through non-parametric learning.
- `generative_pp_VAE/VAE_CEG.py` defines the genrative point process learned through variational learning.
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
