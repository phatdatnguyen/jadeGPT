
# jadeGPT

This is a conversion of [nanoGPT](https://github.com/karpathy/nanoGPT) to Jupyter Notebook and Gradio web ui app. Big thanks to Andrej Karpathy. You should watch his video here: https://www.youtube.com/watch?v=kCc8FmEb1nY

## Gradio web app
![Train tab](/images/ui1.png "Train tab")
![Fine-tune tab](/images/ui2.png "Fine-tune")
![Text generation](/images/ui3.png "Text generation")

## Installation
1. Install [git](https://git-scm.com/)
2. Install [python 3.10](https://www.python.org/downloads/release/python-31010/)
3. Install [Jupyter Notebook](https://jupyter.org/install) or [Gradio](https://gradio.app/quickstart/)
4. Dependencies:
- [pytorch](https://pytorch.org)
- [numpy](https://numpy.org/install/)
- `pip install transformers`
- `pip install datasets`
- `pip install tiktoken`

## Quick guide
In terminal: `git clone https://github.com/phatdatnguyen/jadeGPT`

For Jupyter Notebook
- Open Jupyter Notebook and navigate to the jadeGPT folder
- Use **train-gpt.ipynb** to train a new GPT model using a custom dataset
- Use **finetune-gpt.ipynb** to finetune a checkpoint or a pretrained GPT2 model
- Use **sample-gpt.ipynb** to generate text from a checkpoint
- Use **sample-gpt2.ipynb** to generate text from a pretrained GPT2 model

For Gradio
- Open jadeGPT folder
- In terminal: `python jadegpt_ui.py`
- Use the **Train** tab for training a new GPT model using a custom dataset
- Use the **Fine-tune** tab for fine-tuning a checkpoint or a pretrained GPT2 model
- If you only want to generate text from a checkpoint or a pretrained GPT2 model, use to **Fine-tune** tab to load the model and start generating text
