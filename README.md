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

## Debugging

Model does not create expected correlations in the generated time series. Using web/wiki data from the original paper, http://arxiv.org/abs/1909.13403, we expect to see 7-day correlation spikes in the generated time series, but the model does not recreate those. Attribute distributions look good, but feature generation is not working.

Known differences between this code and the tf1 code that works:
* Variable length sequences are not supported. This means the gen_flag features are not included in training data or generated data.
* Softmax outputs used for later steps in generator instead of taking one-hot representation of the argmax.
* Parameter initialization differences between PyTorch and tf1. Notably, there is no equivalent in PyTorch to the forget_bias parameter in the tf1 LSTMCell that initializes the forget gate bias to 1.
* CPU only
* Additional attributes and attributes use the same input noise vector.

We don't think the above are the cause of the performance differences, but they could be and are potential avenues of investigation.
