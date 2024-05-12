# llm_pytorch
**A large language model in PyTorch (train/run)**


https://colab.research.google.com/github/mareksdfgh/llm_pytorch/blob/main/colab.ipynb



This repository contains the code to train and execute a Large Language Model (LLM).

## Requirements

To run the code, you'll need:

- The required Python libraries listed in `requirements.txt`. You can install them by running the following command:
- `pip install -r requirements.txt`


```bash
  pip install -r requirements.txt
```

## Training the Model

To train the LLM model, run the `train.py` script. Make sure to have a suitable dataset beforehand. Currently, a dataset in text format named `input.txt` is expected.

```bash
python train.py
```



## Executing the Model

Once the model is trained, you can use the `run.py` script to execute it. This script allows you to generate text based on the trained model or perform other tasks the model is designed for. Ensure that the model is properly trained before executing `run.py`.

To execute the model, run the following command:


```bash
python run.py
```
