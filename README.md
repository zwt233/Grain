# Grain: Improving Data Efficiency of Graph Neural Networks via Diversified Influence Maximization.

This repository is the official implementation of GRAIN. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training

To train the model(s) in the paper:
> cd the  “example” directory

>run the python file GRAIN(ball-D).py or GRAIN(NN-D).py

You can also see the experiment results in the notebook:
> cd the  “example” directory

>run the notebook file Test.ipynb

## Results

1. Accuracy comparison:

<img src="accuracy.png" width="80%" height="80%">

2. Active learning comparison:

<img src="active learning.png" width="120%" height="80%">

3. Core-set selection comparison:

<img src="core-set selection.png" width="120%" height="80%">

4. Efficiency comparison:

<img src="speedup.png" width="80%" height="80%">

5. Interpretability:

<img src="interpretability.png" width="80%" height="80%">
