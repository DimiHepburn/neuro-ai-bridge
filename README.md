---

## 🔗 Related Repositories

This repository is one vertex of a three-part research programme. The repositories are deliberately complementary:

| Repository | Question | Focus |
|------------|----------|-------|
| **neuro-ai-bridge** *(this repo)* | *What do brains do?* | Biological learning mechanisms — Hebbian plasticity, spiking neurons, predictive coding, hippocampal memory, biased-competition attention |
| [**llm-interpretability-notes**](https://github.com/DimiHepburn/llm-interpretability-notes) | *What do models do?* | Reverse-engineering transformer internals — residual stream, attention circuits, sparse autoencoders, activation patching |
| [**humanising-ai**](https://github.com/DimiHepburn/humanising-ai) | *What should models do?* | Frameworks and reference implementation for affective computing, theory of mind, dialogue grounding, explainability, and handoff |

**Concrete bridges between the three:**

- The **attention-as-gating** module here provides the neuroscientific grounding for the attention-circuit analyses in [`llm-interpretability-notes`](https://github.com/DimiHepburn/llm-interpretability-notes).
- The **hippocampal memory** module motivates the context-management design in [`humanising-ai/src/dialogue/context_manager.py`](https://github.com/DimiHepburn/humanising-ai/blob/main/src/dialogue/context_manager.py).
- The **predictive processing** module gives a principled account of why explainability methods (SHAP, contrastive explanations) matter: they expose the model's internal predictive model to the user.

---

## 📚 Key References

- Hebb, D.O. (1949). *The Organisation of Behavior*
- Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- McClelland, J.L. et al. (1995). Why there are complementary learning systems in the hippocampus and neocortex
- Rao, R.P.N. & Ballard, D.H. (1999). Predictive coding in the visual cortex
- Buzsáki, G. (2006). *Rhythms of the Brain*

---

## 🚀 Getting Started

```bash
git clone https://github.com/DimiHepburn/neuro-ai-bridge.git
cd neuro-ai-bridge
pip install -r requirements.txt
jupyter notebook notebooks/01_hebbian_learning.ipynb
```

---

## 🤝 Contributing

This is an active personal research project. If you're working on similar ideas at the intersection of neuroscience and AI, feel free to open an issue or reach out — collaboration is always welcome.

---

*Part of a broader research programme on neuroscience-inspired AI and the humanisation of artificial intelligence.*

*See also: [llm-interpretability-notes](https://github.com/DimiHepburn/llm-interpretability-notes) | [humanising-ai](https://github.com/DimiHepburn/humanising-ai)*
