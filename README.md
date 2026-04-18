# 🧠 neuro-ai-bridge

*Bridging the gap between biological neural systems and artificial intelligence architectures.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen?style=flat-square)]()

## Overview

This repository explores how insights from **computational neuroscience** can inform, improve, and inspire modern AI systems. The central thesis is that the brain has already "solved" many of the problems that AI researchers are still grappling with — and that studying biological systems rigorously can give us better inductive biases for building intelligent machines.

The work spans several interconnected themes: how neurons encode and transmit information, how memories are formed and consolidated, how attention and prediction shape perception, and what it would mean for an artificial system to learn the way brains do.

## 🔑 Key Topics & Modules

### 1. Hebbian Learning & Synaptic Plasticity

*"Neurons that fire together, wire together."*

Hebbian learning is one of the oldest and most biologically grounded learning rules. Unlike backpropagation, it is **local** — a synapse updates based only on the activity of the two neurons it connects, with no global error signal required.

This module implements and analyses:

- Classical Hebbian update rules
- Oja's rule (normalised Hebbian learning with weight decay)
- BCM (Bienenstock-Cooper-Munro) theory for synaptic modification
- Spike-Timing-Dependent Plasticity (STDP)

```python
import torch

def hebbian_update(weights, pre, post, lr=0.01):
    """
    Classic Hebbian weight update.
    Δw_ij = η * x_i * y_j
    """
    delta_w = lr * torch.outer(post, pre)
    return weights + delta_w

def oja_rule(weights, pre, post, lr=0.01):
    """
    Oja's normalised Hebbian rule — prevents weight explosion.
    Δw_ij = η * y_j * (x_i - y_j * w_ij)
    """
    delta_w = lr * (torch.outer(post, pre) - torch.outer(post * post, weights.sum(0)))
    return weights + delta_w
```

### 2. Spiking Neural Networks (SNNs)

Biological neurons communicate via discrete **spikes** (action potentials), not continuous-valued activations. SNNs are the closest artificial analogue to real neural computation. They are:

- Natively **temporal** — information is encoded in spike timing, not just rate
- Potentially highly **energy-efficient** on neuromorphic hardware (Intel Loihi, IBM TrueNorth)
- A natural fit for **event-driven** sensory processing

This module uses [snnTorch](https://github.com/jeshraghian/snntorch) to implement and experiment with:

- Leaky Integrate-and-Fire (LIF) neurons
- Rate coding vs. temporal coding experiments
- STDP-trained SNN classifiers on MNIST and N-MNIST (neuromorphic dataset)

```python
import snntorch as snn
import torch.nn as nn

class SpikingNet(nn.Module):
    def __init__(self):
        super().__init__()
        beta = 0.95  # membrane potential decay rate
        self.fc1 = nn.Linear(784, 256)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(256, 10)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x, mem1, mem2):
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        return spk2, mem1, mem2
```

### 3. Predictive Processing & the Free Energy Principle

Karl Friston's **Free Energy Principle** proposes that the brain is fundamentally a prediction machine — constantly generating models of the world and updating them based on prediction errors. This has profound implications for AI:

- **Top-down predictions** constrain bottom-up sensory processing
- Learning is minimising **surprise** (technically: variational free energy)
- Perception and action are unified under a single objective

This module implements:

- A simple hierarchical predictive coding network
- Variational autoencoders (VAEs) interpreted through a free-energy lens
- Comparisons between predictive coding and standard backprop networks on perceptual tasks

### 4. Hippocampal Memory & Complementary Learning Systems

The hippocampus and neocortex implement two distinct learning systems:

| System | Speed | Capacity | Function |
|--------|-------|----------|----------|
| Hippocampus | Fast (one-shot) | Limited | Episodic memory, rapid binding |
| Neocortex | Slow (gradual) | Large | Statistical regularities, semantic memory |

This dual-system architecture is the inspiration behind **experience replay** in deep RL. This module:

- Implements a simplified hippocampal episodic buffer with fast one-shot storage
- Demonstrates how replay prevents catastrophic forgetting in continual learning
- Connects to modern work on memory-augmented neural networks (NTM, DNC)

### 5. Attention as Selective Neural Gating

Transformer attention mechanisms bear a striking resemblance to **biased competition** models of selective attention in primate visual cortex. This module draws connections between:

- The query-key-value attention formulation and top-down attentional biasing
- Predictive attention (anticipating where to look based on context)
- Multi-head attention as an analogue to different attentional spotlights

## 📂 Repository Structure
neuro-ai-bridge/
├── notebooks/
│   ├── 01_hebbian_learning.ipynb
│   ├── 02_spiking_neural_networks.ipynb
│   ├── 03_predictive_processing.ipynb
│   ├── 04_hippocampal_memory.ipynb
│   └── 05_attention_as_gating.ipynb
├── src/
│   ├── plasticity/
│   │   ├── hebbian.py
│   │   ├── stdp.py
│   │   └── bcm.py
│   ├── snn/
│   │   ├── lif_neuron.py
│   │   └── snn_classifier.py
│   ├── predictive_coding/
│   │   ├── pc_network.py
│   │   └── free_energy.py
│   └── memory/
│       ├── episodic_buffer.py
│       └── replay.py
├── data/
├── results/
├── requirements.txt
└── README.md

## 🔗 Related Repositories

This repository is one vertex of a three-part research programme. The repositories are deliberately complementary:

| Repository | Question | Focus |
|------------|----------|-------|
| **neuro-ai-bridge** *(this repo)* | *What do brains do?* | Biological learning mechanisms — Hebbian plasticity, spiking neurons, predictive coding, hippocampal memory, biased-competition attention |
| [**llm-interpretability-notes**](https://github.com/DimiHepburn/llm-interpretability-notes) | *What do models do?* | Reverse-engineering transformer internals — residual stream, attention circuits, sparse autoencoders, activation patching |
| [**humanising-ai**](https://github.com/DimiHepburn/humanising-ai) | *What should models do?* | Frameworks and reference implementation for affective computing, theory of mind, dialogue grounding, explainability, and handoff |

**Concrete bridges between the three:**

- The **attention-as-gating** module here provides the neuroscientific grounding for the attention-circuit analyses in [llm-interpretability-notes](https://github.com/DimiHepburn/llm-interpretability-notes).
- The **hippocampal memory** module motivates the context-management design in [humanising-ai/src/dialogue/context_manager.py](https://github.com/DimiHepburn/humanising-ai/blob/main/src/dialogue/context_manager.py).
- The **predictive processing** module gives a principled account of why explainability methods (SHAP, contrastive explanations) matter: they expose the model's internal predictive model to the user.

## 📚 Key References

- Hebb, D.O. (1949). *The Organisation of Behavior*
- Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- McClelland, J.L. et al. (1995). Why there are complementary learning systems in the hippocampus and neocortex
- Rao, R.P.N. & Ballard, D.H. (1999). Predictive coding in the visual cortex
- Buzsáki, G. (2006). *Rhythms of the Brain*

## 🚀 Getting Started

```bash
git clone https://github.com/DimiHepburn/neuro-ai-bridge.git
cd neuro-ai-bridge
pip install -r requirements.txt
jupyter notebook notebooks/01_hebbian_learning.ipynb
```

## 🤝 Contributing

This is an active personal research project. If you're working on similar ideas at the intersection of neuroscience and AI, feel free to open an issue or reach out — collaboration is always welcome.

---

*Part of a broader research programme on neuroscience-inspired AI and the humanisation of artificial intelligence.*

*See also: [llm-interpretability-notes](https://github.com/DimiHepburn/llm-interpretability-notes) | [humanising-ai](https://github.com/DimiHepburn/humanising-ai)*
