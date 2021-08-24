# Explainer-Critic 

## Idea
One neural network - the _explainer_ - learns a classification task. 
We then use a saliency method to explain its decisions. 
A second neural network - the _critic_ - learns the classification task, 
and uses the explanation to improve its learning. 
We say that an explanation was successful if it speeds up the learner's training. 
The critic's end-of-training loss is used as a loss function for the explainer's explanation quality. 
This captures the intuitive notion that a teacher is good at explaining 
if their explanations help a student to learn quickly. 

Here you can find a [formalization and more detailed description](https://hackmd.io/zEC0IZk5TVyVyysqPqDp2A?both).

## Usage

Run `python3 main.py -h` to show available options to run.

## Setup

* `pip install -r requirements.txt`
* If CUDA is too old, check out which previous versions of torch to install [here](https://pytorch.org/get-started/previous-versions/).
    * get the newest pytorch version that still works with your CUDA version.
    * for example, for CUDA version 10.1, do 

```
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```