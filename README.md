# Conditional Generative Modeling is All You Need for Marked Temporal Point Processes

Recent advancements in generative modeling have made it possible to generate high-quality content from context information, but a key question remains: how to teach models to know when to generate content? To answer this question, this study proposes a novel event generative model that draws its statistical intuition from marked temporal point processes, and offers a clean, flexible, and computationally efficient solution for a wide range of applications involving multi-dimensional marks. We aim to capture the distribution of the point process without explicitly specifying the conditional intensity or probability density. Instead, we use a conditional generator that takes the history of events as input and generates the high-quality subsequent event that is likely to occur given the prior observations. The proposed framework offers a host of benefits, including exceptional efficiency in learning the model and generating samples, as well as considerable representational power to capture intricate dynamics in multi- or even high-dimensional event space. Our numerical results demonstrate superior performance compared to other state-of-the-art baselines.

## Citation

```
@article{dong2023conditional,
  title={Conditional Generative Modeling is All You Need for Marked Temporal Point Processes},
  author={Dong, Zheng and Fan, Zekai and Zhu, Shixiang},
  journal={arXiv preprint arXiv:2305.12569},
  year={2023}
}
```
