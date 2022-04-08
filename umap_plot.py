import torch
from torch import Tensor
from utils import FastMNIST, set_device
import umap
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="paper", style="white")

set_device()
mnist = FastMNIST('./data', train=True, download=True)

reducer = umap.UMAP(random_state=42)
data: Tensor = torch.flatten(mnist.data, start_dim=1, end_dim=3)

print('computing embedding...')
embedding = reducer.fit_transform(data)
print('finished computing embedding')

fig, ax = plt.subplots(figsize=(12, 10))
color = mnist.targets
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)

plt.show()