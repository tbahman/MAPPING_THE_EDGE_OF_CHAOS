# Mapping the Edge of Chaos

**Fractal-Like Boundaries in The Trainability of Decoder-Only Transformer Models**

This repository contains code from the study *Mapping the Edge of Chaos: Fractal-Like Boundaries in The Trainability of Decoder-Only Transformer Models*. The project explores how subtle adjustments in hyperparameters—particularly learning rates for attention versus fully connected layers—can push training dynamics from convergence into divergence, revealing intricate fractal-like boundaries.

## Overview

Training large-scale language models is a delicate balancing act. Drawing inspiration from fractal geometry, this work investigates the “edge of chaos” where the convergence behavior of a decoder-only transformer is highly sensitive to hyperparameter choices. By mapping the hyperparameter landscape with a novel convergence measure, the experiments uncover self-similar, chaotic boundaries that repeat across scales.

<img src="etc/zoomed_10.png" alt="mu-mu" style="width:40%; height:40%;">   <img src="etc/zoomed_11.png" alt="mu-mu" style="width:40%; height:40%;">

<div style="position: relative; width: 400px;">
  <img src="etc/zoomed_10.png" alt="First image" style="width: 40%;">
  <img src="etc/zoomed_11.png" alt="Second image" style="position: absolute; bottom: 0; right: -10; width: 40%; z-index: 10;">
</div>


## Summary

The paper demonstrates that:
- The boundary between stable (convergent) and unstable (divergent) training regimes exhibits chaotic characteristics.
- A consistent convergence measure is defined—based on the loss function’s behavior to evaluate training stability.
- Experiments reveal that slight adjustments in learning rates (for attention and fully connected layers) lead to a complex partitioning of the hyperparameter space.

## LLM Architecture 

- Using a decoder-only transformer model (with 95,973 trainable parameters) trained on character-level data from Shakespeare’s works, the study visualizes these boundaries.
