# Equitable and Optimal Transport
Code of the [paper](https://arxiv.org/pdf/2006.07260.pdf) by Meyer Scetbon, Laurent Meunier, Jamal Atif and Marco Cuturi.

## A New Fair Optimal Transport Problem Between Multiple Agents
In this work, we introduce an extension of the Optimal Transport, namely the Equitable Optimal Transport (EOT) problem, when multiple costs are involved. Considering each cost as an agent, we aim to share equally between agents the work of transporting one distribution to another. In the following figure, we illustrate how our method split the transportation task between 3 given agents.
![figure](results/primal_W.png)
<img src="primal_W.png" alt="drawing" width="200"/>

To be able to compute EOT, we propose to regularize it by regularizing the objective with an entropic term. Such regularization leads to a Sinkhorn-like algorithm which manages to compute efficiently an approximation of EOT. 

This repository contains a Python implementation of the algorithms presented in the [paper](https://arxiv.org/pdf/2006.07260.pdf).
