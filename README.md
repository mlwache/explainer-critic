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

## To Do

* [x] train on small subset (maybe 500 samples) of MNIST only, to make developing/debugging quicker.
* [x] compute the input gradient of a some test images, and show them
* [ ] add logging
* [ ] finish building the "skeleton" (method stubs)
* [ ] fill them.  
* [ ] write a few simple tests  
* [ ] debug