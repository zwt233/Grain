# Grain: Improving Data Efficiency of Graph Neural Networks via Diversified Influence Maximization.

This repository is the official implementation of GRAIN. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training

To train the model(s) in the paper:


> cd the  “example” data

>run the python file GRAIN(ball-D).py or GRAIN(NN-D).py


## Results

1. Accuracy comparison:

<img src="accuracy.png" width="80%" height="80%">

2. Active learning comparison:

<img src="active learning.png" width="80%" height="80%">

3. Core-set selection comparison:

<img src="core-set selection.png" width="80%" height="80%">

4. Efficiency comparison on GPU:

<img src="efficiency_gpu.png" width="80%" height="80%">

5. Efficiency comparison on CPU:

<img src="efficiency_cpu.png" width="80%" height="80%">

6. Interpretability:

<img src="interpretability.png" width="80%" height="80%">

7. Ablation study:

<img src="ablation.png" width="80%" height="80%">

8. Generalization:

<img src="generalization.png" width="80%" height="80%">

## Cite

If you use Grain in a scientific publication, we would appreciate citations to the following paper:
```
@article{zhang2021grain,
  title={GRAIN: improving data efficiency of gra ph neural networks via diversified in fluence maximization},
  author={Zhang, Wentao and Yang, Zhi and Wang, Yexin and Shen, Yu and Li, Yang and Wang, Liang and Cui, Bin},
  journal={Proceedings of the VLDB Endowment},
  volume={14},
  number={11},
  pages={2473--2482},
  year={2021},
  publisher={VLDB Endowment}
}
```


