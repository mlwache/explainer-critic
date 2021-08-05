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

* Run `python3 main.py -h` to show available options to run.