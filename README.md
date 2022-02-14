# doppelganger-torch
PyTorch implementation of DoppelGANger, work in progress. **Does not produce good output at this time!**

Based on https://github.com/fjxmlzn/DoppelGANger, but converted to PyTorch.

## Installation

```
pip install -r requirements.txt
```

(Recommended to install within a virtual environment of some type.)

## Testing

Very simple tests are currently in `test_doppelganger.py`. These mostly run 1
epoch on some random data to make sure tensor shapes align properly.

```
pytest test_doppelganger.py
```

## Differences to original paper/tf1 code

Known differences between this code and the tf1 code:
* Variable length sequences are not supported. This means the gen_flag features are not included in training data or generated data.
* Softmax outputs used for later steps in generator instead of taking one-hot representation of the argmax.
* Parameter initialization differences between PyTorch and tf1. Notably, there is no equivalent in PyTorch to the forget_bias parameter in the tf1 LSTMCell that initializes the forget gate bias to 1.
* CPU only
* Additional attributes and attributes use the same input noise vector.
