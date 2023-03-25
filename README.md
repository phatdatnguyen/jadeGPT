
# jadeGPT

This is a conversion of [nanoGPT](https://github.com/karpathy/nanoGPT) to Jupyter Notebook and Gradio web ui app. Big thanks to Andrej Karpathy. You should watch his video here: https://www.youtube.com/watch?v=kCc8FmEb1nY

## Install
1. Install [Jupyter Notebook](https://jupyter.org/install) or [Gradio](https://gradio.app/quickstart/)
2. Dependencies:
- [pytorch](https://pytorch.org)
- [numpy](https://numpy.org/install/)
- `pip install transformers`
- `pip install datasets`
- `pip install tiktoken`

## Quick guide
For Jupyter Notebook
- use **train-gpt.ipynb** to train a new GPT model using a custom dataset
- use **finetune-gpt.ipynb** to finetune a checkpoint or a pretrained GPT2 model
- use **sample-gpt.ipynb** to generate text from a checkpoint
- use **sample-gpt2.ipynb** to generate text from a pretrained GPT2 model

For Gradio
- `python jadegpt_ui.py`
- use the **Train** tab for training a new GPT model using a custom dataset
- use the **Fine-tune** tab for fine-tuning a checkpoint or a pretrained GPT2 model
