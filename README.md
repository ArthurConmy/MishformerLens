# MishformerLens

> MishformerLens intends to be a drop-in replacement for TransformerLens that AST patches HuggingFace Transformers rather than implementing a custom, numerically inaccurate Transformer architecture.

MishformerLens is currently highly experimental and at version 0.0.x.

Status as of 5th October: https://www.diffchecker.com/TaW9IAhJ shows the difference between `https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Exploratory_Analysis_Demo.ipynb` and `MishformerLens/mishformer_lens/mishformer_lens_expoloratory_analysis_demo.py` -- it's basically just formatting.

Note that we only have support for GPT-2, and no fold LN etc. stuff, just `from_pretrained_no_preprocessing` basically. It should be pretty easy to add fold LN etc. (with a small risk of numerical problems), and will be a lot harder to add support for every single model family.

TODO(v0.1): write this up in full

# Roadmap

v0.1: make this usable for most TransformerLens models, including everything upstreamed to TL.
v1: PyPI, full testing, library ready for development.

# Installation notes

TODO(v0.1): clean up

```bash
# Essential installs:
#
# Install transformers==4.45.1
# pip install this fork of TransformerLens 2.7.1: https://github.com/ArthurConmy/TransformerLens/tree/mishformer-lens-changes  # TODO(v0.1): upstream TransformerLens changes
# Install https://github.com/google-deepmind/mishax at commit hash 617972a2f83f14b3b76288477974d95563fe5e7d
#
# Lower priority installs, but still nice-to-haves:
#
# Install IPython
# Install plotly==5.24.1
# Install nbformat>=4.20.0
# Install this repo
```

N.B. I may upstream some changes so the above could be inaccurate
