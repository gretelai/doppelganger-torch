from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import tensorboard
from tqdm import tqdm

# Continuous representations
# * original - arbitrary values, no additional attributes
# * feature scaled - [0,1] or [-1,1], no additional attributes
# * example scaled - [0,1] or [-1,1] achieving min and max for each time series, additional attributes with midpoing and half range
#
# Discrete representations
# * strings? - 1 dim per discrete feature, stored as categorical/string
# * original - 1 dim per discrete feature, stored as index (float for now)
# * one-hot - X dims per discrete feature, stored as one-hot (float for now, eventually long?)
# * probabilistic - X dims per discrete feature, soft max (float)
#


def autocovariance(a):
    """Compute autocovariance of multiple time series.

    Args:

        a: Time series with 1 channel. Accepts 2-d numpy array of examples x
            time, or 3-d numpy array with
            examples x time x channel where the last dimension is of size 1.
    """
    if len(a.shape) == 3:
        if a.shape[2] != 1:
            raise RuntimeError(
                f"Unexpected shape={a.shape} for autocovariance calculation."
            )
        a = a.reshape(a.shape[0], a.shape[1])

    s = np.zeros(2 * a.shape[1] - 1)
    for i in range(a.shape[0]):
        ts = (
            a[i, :]
            - a[
                i,
                :,
            ].mean()
        )
        ss = np.correlate(ts, ts, mode="full")
        s += ss

    s = s[s.size // 2 :]
    counts = np.arange(len(s), 0, -1) * a.shape[0]
    s /= counts
    s /= a.var()

    return s


class OutputType(str, Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class Normalization(str, Enum):
    ZERO_ONE = "zero_one"
    MINUSONE_ONE = "minusone_one"


@dataclass(frozen=True)
class Output:
    """Metadata for a variable, used for features, attributes, and additional_attributes."""

    name: str

    # TODO: better way to always have dim=1 in ContinuousOutput?
    def get_dim(self):
        return 1


@dataclass(frozen=True)
class DiscreteOutput(Output):
    """Discrete variable metadata."""

    dim: int

    def get_dim(self):
        return self.dim


@dataclass(frozen=True)
class ContinuousOutput(Output):
    """Continuous variable metadata."""

    normalization: Normalization
    global_min: float
    global_max: float
    is_feature_normalized: bool
    is_example_normalized: bool


def prepare_data(
    original_attributes: np.ndarray,
    attribute_types: List[OutputType],
    original_features: np.ndarray,
    feature_types: List[OutputType],
    normalization: Normalization = Normalization.ZERO_ONE,
    is_feature_normalized: bool = True,
    is_example_normalized: bool = True,
):
    def make_output(
        index: int,
        t: OutputType,
        data: np.ndarray,
        is_feature_normalized: bool,
        is_example_normalized: bool,
    ) -> Output:
        if t == OutputType.CONTINUOUS:
            output = ContinuousOutput(
                name="a" + str(index),
                normalization=normalization,
                global_min=np.min(data),
                global_max=np.max(data),
                is_feature_normalized=is_feature_normalized,
                is_example_normalized=is_example_normalized,
            )
        elif t == OutputType.DISCRETE:
            output = DiscreteOutput(
                name="a" + str(index),
                dim=1 + np.int32(np.max(data)),
            )
        else:
            raise RuntimeError(f"Unknown output type={t}")
        return output

    attribute_outputs = [
        make_output(
            index,
            t,
            original_attributes[:, index],
            is_feature_normalized=is_feature_normalized,
            is_example_normalized=False,
        )
        for index, t in enumerate(attribute_types)
    ]
    feature_outputs = [
        make_output(
            index,
            t,
            original_features[:, :, index],
            is_feature_normalized=is_feature_normalized,
            is_example_normalized=is_example_normalized,
        )
        for index, t in enumerate(feature_types)
    ]

    attributes, _, _ = transform(
        original_attributes, attribute_outputs, variable_dim_index=1
    )
    features, additional_attribute_outputs, additional_attributes = transform(
        original_features,
        feature_outputs,
        variable_dim_index=2,
    )

    return DGData(
        attributes=attributes,
        additional_attributes=additional_attributes,
        features=features,
        attribute_outputs=attribute_outputs,
        additional_attribute_outputs=additional_attribute_outputs,
        feature_outputs=feature_outputs,
    )


def rescale(
    original: np.ndarray,
    normalization: Normalization,
    global_min: Union[float, np.ndarray],
    global_max: Union[float, np.ndarray],
) -> np.ndarray:
    if normalization == Normalization.ZERO_ONE:
        # TODO: why is pylance/pytype not happy with this?
        return (original - global_min) / (global_max - global_min)
    elif normalization == Normalization.MINUSONE_ONE:
        return (2.0 * (original - global_min) / (global_max - global_min + 1e-6)) - 1.0


def rescale_inverse(
    transformed: np.ndarray,
    normalization: Normalization,
    global_min: Union[float, np.ndarray],
    global_max: Union[float, np.ndarray],
) -> np.ndarray:
    if normalization == Normalization.ZERO_ONE:
        return transformed * (global_max - global_min) + global_min
    elif normalization == Normalization.MINUSONE_ONE:
        return ((transformed + 1) / 2) * (global_max - global_min) + global_min


def transform(
    original_data: np.ndarray, outputs: List[Output], variable_dim_index: int
):
    # TODO: better way to move additional attributes around while still keeping
    # this transform a reusable function?

    # TODO: is there a better/more efficient way to do this wrt numpy?

    additional_attribute_outputs = []
    additional_attribute_parts = []
    parts = []
    for index, output in enumerate(outputs):
        # NOTE: isinstance(output, DiscreteOutput) does not work consistently
        #       with all import styles in jupyter notebooks, using string
        #       comparison instead.
        if "DiscreteOutput" in str(output.__class__):
            # TODO: separate float and long tensors
            if variable_dim_index == 1:
                indices = original_data[:, index].astype(int)
            elif variable_dim_index == 2:
                indices = original_data[:, :, index].astype(int)
            else:
                raise RuntimeError(
                    f"Unsupported variable_dim_index={variable_dim_index}"
                )

            if variable_dim_index == 1:
                b = np.zeros((len(indices), output.dim))
                b[np.arange(len(indices)), indices] = 1
            elif variable_dim_index == 2:
                b = np.zeros((indices.shape[0], indices.shape[1], output.dim))
                # From https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
                def all_idx(idx, axis):
                    grid = np.ogrid[tuple(map(slice, idx.shape))]
                    grid.insert(axis, idx)
                    return tuple(grid)

                b[all_idx(indices, axis=2)] = 1

            parts.append(b)
        elif "ContinuousOutput" in str(output.__class__):
            if variable_dim_index == 1:
                raw = original_data[:, index]
            elif variable_dim_index == 2:
                raw = original_data[:, :, index]
            else:
                raise RuntimeError(
                    f"Unsupported variable_dim_index={variable_dim_index}"
                )

            if output.is_feature_normalized:
                feature_scaled = rescale(
                    raw, output.normalization, output.global_min, output.global_max
                )
            else:
                feature_scaled = raw

            if output.is_example_normalized:
                if variable_dim_index != 2:
                    raise RuntimeError(
                        "is_example_normalized only applies to features where the data has 3 dimensions"
                    )
                # TODO: midpoint and half range also need to be added as
                # additional attribute, but happens separately outside this
                # function for now
                mins = np.min(feature_scaled, axis=1)
                maxes = np.max(feature_scaled, axis=1)

                # Handle additional attributes
                # TODO: there's an assumption here that if is_example_normalized is True, then
                # either is_feature_normalized is True OR the data was already scaled to [0,1] or [-1,1]
                additional_attribute_outputs.append(
                    ContinuousOutput(
                        name=output.name + "_midpoint",
                        normalization=output.normalization,
                        # TODO do something about global min and max?
                        global_min=0.0
                        if output.normalization == Normalization.ZERO_ONE
                        else -1.0,
                        global_max=1.0,
                        is_feature_normalized=False,  # TODO should this be True?
                        is_example_normalized=False,
                    )
                )
                additional_attribute_outputs.append(
                    ContinuousOutput(
                        name=output.name + "_half_range",
                        normalization=Normalization.ZERO_ONE,
                        global_min=0.0,
                        global_max=1.0,
                        is_feature_normalized=False,  # TODO: should this be True?
                        is_example_normalized=False,
                    )
                )

                additional_attribute_parts.append(
                    ((mins + maxes) / 2).reshape(mins.shape[0], 1)
                )
                additional_attribute_parts.append(
                    ((maxes - mins) / 2).reshape(mins.shape[0], 1)
                )

                mins = np.broadcast_to(
                    mins.reshape(mins.shape[0], 1),
                    (mins.shape[0], feature_scaled.shape[1]),
                )
                maxes = np.broadcast_to(
                    maxes.reshape(maxes.shape[0], 1),
                    (mins.shape[0], feature_scaled.shape[1]),
                )

                scaled = rescale(feature_scaled, output.normalization, mins, maxes)
            else:
                scaled = feature_scaled

            if variable_dim_index == 1:
                scaled = scaled.reshape(original_data.shape[0], 1)
            elif variable_dim_index == 2:
                scaled = scaled.reshape(
                    original_data.shape[0], original_data.shape[1], 1
                )
            parts.append(scaled)
        else:
            raise RuntimeError(f"Unsupported output type, class={type(output)}'")

    additional_attributes = None
    if additional_attribute_parts:
        additional_attributes = np.concatenate(additional_attribute_parts, axis=1)
    return (
        np.concatenate(parts, axis=variable_dim_index),
        additional_attribute_outputs,
        additional_attributes,
    )


def inverse_transform(
    transformed_data: np.ndarray,
    outputs: List[Output],
    variable_dim_index: int,
    additional_attributes: np.ndarray = None,
    additional_attribute_outputs: List[Output] = None,
):
    # TODO: less messy handling of additional attributes for per example scaling?
    parts = []
    transformed_index = 0
    additional_attribute_index = 0
    for index, output in enumerate(outputs):
        if "DiscreteOutput" in str(output.__class__):
            if variable_dim_index == 1:
                onehot = transformed_data[
                    :, transformed_index : (transformed_index + output.dim)
                ]
            elif variable_dim_index == 2:
                onehot = transformed_data[
                    :, :, transformed_index : (transformed_index + output.dim)
                ]
            else:
                raise RuntimeError(
                    f"Unsupported variable_dim_index={variable_dim_index}"
                )
            indices = np.argmax(onehot, axis=variable_dim_index)

            target_shape = list(transformed_data.shape)
            target_shape[-1] = 1
            indices = indices.reshape(target_shape)

            parts.append(indices)
            transformed_index += output.dim
        elif "ContinuousOutput" in str(output.__class__):
            if variable_dim_index == 1:
                transformed = transformed_data[:, transformed_index]
            elif variable_dim_index == 2:
                transformed = transformed_data[:, :, transformed_index]
            else:
                raise RuntimeError(
                    f"Unsupported variable_dim_index={variable_dim_index}"
                )

            if output.is_example_normalized:
                if variable_dim_index != 2:
                    raise RuntimeError(
                        "is_example_normalized only applies to features where the data has 3 dimensions"
                    )

                if (
                    additional_attributes is None
                    or additional_attribute_outputs is None
                ):
                    raise RuntimeError(
                        "Must provide additional_attributes and additional_attribute_outputs if is_example_normalized=True"
                    )

                midpoint = additional_attributes[:, additional_attribute_index]
                half_range = additional_attributes[:, additional_attribute_index + 1]
                # TODO: do the additional attributes also need to be scaled
                # correctly? seems messy if normalization is ZERO_ONE so the
                # half range should be in [0,0.5]
                additional_attribute_index += 2

                mins = midpoint - half_range
                maxes = midpoint + half_range
                mins = np.expand_dims(mins, 1)
                maxes = np.expand_dims(maxes, 1)

                example_scaled = rescale_inverse(
                    transformed,
                    normalization=output.normalization,
                    global_min=mins,
                    global_max=maxes,
                )
            else:
                example_scaled = transformed

            if output.is_feature_normalized:
                original = rescale_inverse(
                    example_scaled,
                    output.normalization,
                    output.global_min,
                    output.global_max,
                )
            else:
                original = example_scaled

            target_shape = list(transformed_data.shape)
            target_shape[-1] = 1
            original = original.reshape(target_shape)

            parts.append(original)
            transformed_index += 1
        else:
            raise RuntimeError(f"Unsupported output type, class={type(output)}'")

    return np.concatenate(parts, axis=variable_dim_index)


@dataclass(frozen=True)
class DGData:
    """Collect transformed data and info for reverse transforms in one place.

    Discrete variables are one-hot encoded.

    Continuous variables are feature and/or example scaled based on Output settings.
    """

    # TODO: Make this entirely internal, so interface accepts unscaled values
    # (exactly what will be returned by generate function) and all
    # transformation are with the DGTorch class.
    attributes: np.ndarray
    additional_attributes: Union[np.ndarray, None]
    features: np.ndarray
    attribute_outputs: List[Output]
    additional_attribute_outputs: List[Output]
    feature_outputs: List[Output]


class Merger(torch.nn.Module):
    """Merge several layers into one concatenated layer."""

    def __init__(self, modules, dim_index: int):
        super(Merger, self).__init__()
        if isinstance(modules, torch.nn.ModuleList):
            self.layers = modules
        else:
            self.layers = torch.nn.ModuleList(modules)

        self.dim_index = dim_index

    def forward(self, input):
        return torch.cat([m(input) for m in self.layers], dim=self.dim_index)


class OutputDecoder(torch.nn.Module):
    """Decoder to produce continuous or discrete output values as needed."""

    def __init__(self, input_dim: int, outputs: List[Output], dim_index: int):
        super(OutputDecoder, self).__init__()
        if outputs is None or len(outputs) == 0:
            raise RuntimeError("OutputDecoder received no outputs")

        self.dim_index = dim_index
        self.generators = torch.nn.ModuleList()

        for output in outputs:
            if "DiscreteOutput" in str(output.__class__):
                self.generators.append(
                    torch.nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "linear",
                                    torch.nn.Linear(input_dim, output.get_dim()),
                                ),
                                ("softmax", torch.nn.Softmax(dim=dim_index)),
                            ]
                        )
                    )
                )
            elif "ContinuousOutput" in str(output.__class__):
                if output.normalization == Normalization.ZERO_ONE:
                    normalizer = torch.nn.Sigmoid()
                elif output.normalization == Normalization.MINUSONE_ONE:
                    normalizer = torch.nn.Tanh()
                else:
                    raise RuntimeError(
                        f"Unsupported normalization='{output.normalization}'"
                    )
                self.generators.append(
                    torch.nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "linear",
                                    torch.nn.Linear(input_dim, output.get_dim()),
                                ),
                                ("normalization", normalizer),
                            ]
                        )
                    )
                )
            else:
                raise RuntimeError(f"Unsupported output type, class={type(output)}'")

    def forward(self, input):
        # TODO: consider returning tuple instead of concatenating?
        # TODO: return separate long and float tensors?
        outputs = [generator(input) for generator in self.generators]
        merged = torch.cat(outputs, dim=self.dim_index)
        return merged


class SelectLastCell(torch.nn.Module):
    """Select only last layer's hidden output from LSTM module."""

    def forward(self, x):
        out, _ = x
        return out


class Generator(torch.nn.Module):
    def __init__(
        self,
        attribute_outputs: List[Output],
        additional_attribute_outputs: Union[List[Output], None],
        feature_outputs: List[Output],
        max_sequence_len: int,
        sample_len: int,
        attribute_noise_dim: int,
        feature_noise_dim: int,
        attribute_num_units: int,
        attribute_num_layers: int,
        feature_num_units: int,
        feature_num_layers: int,
    ):
        super(Generator, self).__init__()
        assert max_sequence_len % sample_len == 0

        self.sample_len = sample_len
        self.max_sequence_len = max_sequence_len

        self.attribute_gen = self._make_attribute_generator(
            attribute_outputs,
            attribute_noise_dim,
            attribute_num_units,
            attribute_num_layers,
        )

        # TODO: can we get the width directly instead of assuming internal representation?
        attribute_dim = sum(output.get_dim() for output in attribute_outputs)

        if additional_attribute_outputs:
            self.additional_attribute_gen = self._make_attribute_generator(
                additional_attribute_outputs,
                attribute_noise_dim + attribute_dim,
                attribute_num_units,
                attribute_num_layers,
            )
            additional_attribute_dim = sum(
                output.get_dim() for output in additional_attribute_outputs
            )
        else:
            self.additional_attribute_gen = None
            additional_attribute_dim = 0

        self.feature_gen = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "lstm",
                        torch.nn.LSTM(
                            attribute_dim
                            + additional_attribute_dim
                            + feature_noise_dim,
                            feature_num_units,
                            feature_num_layers,
                            batch_first=True,
                        ),
                    ),
                    ("selector", SelectLastCell()),
                    (
                        "merger",
                        Merger(
                            [
                                OutputDecoder(
                                    feature_num_units, feature_outputs, dim_index=2
                                )
                                for _ in range(self.sample_len)
                            ],
                            dim_index=2,
                        ),
                    ),
                ]
            )
        )

    def _make_attribute_generator(
        self, outputs: List[Output], input_dim: int, num_units: int, num_layers: int
    ):
        seq = []
        last_dim = input_dim
        for _ in range(num_layers):
            seq.append(torch.nn.Linear(last_dim, num_units))
            seq.append(torch.nn.ReLU())
            seq.append(torch.nn.BatchNorm1d(num_units))
            last_dim = num_units

        seq.append(OutputDecoder(last_dim, outputs, dim_index=1))
        return torch.nn.Sequential(*seq)

    def forward(self, attribute_noise: torch.Tensor, feature_noise: torch.Tensor):

        attributes = self.attribute_gen(attribute_noise)

        if self.additional_attribute_gen:
            # TODO: use separate noise inputs for additional attributes to match
            # tf1 code.

            # detach() should be equivalent to stop_gradient used in tf1 code.
            attributes_no_gradient = attributes.detach()
            additional_attribute_gen_input = torch.cat(
                (attributes_no_gradient, attribute_noise), dim=1
            )

            additional_attributes = self.additional_attribute_gen(
                additional_attribute_gen_input
            )
        else:
            additional_attributes = None

        # Reshape and expand attributes to match features
        if self.additional_attribute_gen:
            combined_attributes = torch.cat((attributes, additional_attributes), dim=1)
        else:
            combined_attributes = attributes

        # Use detach() to stop gradient flow
        combined_attributes_no_gradient = combined_attributes.detach()

        reshaped_attributes = torch.reshape(
            combined_attributes_no_gradient, (combined_attributes.shape[0], 1, -1)
        )
        reshaped_attributes = reshaped_attributes.expand(-1, feature_noise.shape[1], -1)

        feature_gen_input = torch.cat((reshaped_attributes, feature_noise), 2)

        features = self.feature_gen(feature_gen_input)

        # TODO: is max_sequence_len always the right thing to use here when we
        # have variable length sequences?
        features = torch.reshape(
            features, (features.shape[0], self.max_sequence_len, -1)
        )
        if self.additional_attribute_gen:
            return attributes, additional_attributes, features
        else:
            return attributes, features


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim: int, num_layers: int = 5, num_units: int = 200):
        super(Discriminator, self).__init__()

        seq = []
        last_dim = input_dim
        for _ in range(num_layers):
            seq.append(torch.nn.Linear(last_dim, num_units))
            seq.append(torch.nn.ReLU())
            last_dim = num_units

        seq.append(torch.nn.Linear(last_dim, 1))

        self.seq = torch.nn.Sequential(*seq)

    def forward(self, input: torch.Tensor):
        return self.seq(input)


def interpolate(x1, x2, alpha):
    """Interpolate between 2 tensors.

    Returns x1 + alpha * (x2 - x1)
    """
    diff = x2 - x1
    expanded_dims = [1 for _ in diff.shape]
    expanded_dims[0] = -1
    reshaped_alpha = alpha.reshape(expanded_dims).expand(diff.shape)

    return x1 + reshaped_alpha * diff


def apply_named(m: torch.nn.Module, prefix: str, func):
    """Equivalent to Module.apply, but also provides access to the submodule names."""
    func(m, prefix)

    for name, child in m.named_children():
        apply_named(child, prefix + "." + name, func)


def log_weights(m: torch.nn.Module, tb_writer: tensorboard.SummaryWriter, global_step):
    """Log weight histograms to tensorboard.

    Parameters for m and all children will be logged to tensorboard, including
    submodule names as prefixes to distinguish different layers.
    """

    def f(a, prefix):
        for name, param in a.named_parameters(recurse=False):
            tb_writer.add_histogram(
                "weights/" + prefix + "." + name, param, global_step
            )

    apply_named(m, "", f)


class DGTorch:
    """
    DoppelGANger model.
    """

    def __init__(
        self,
        attribute_outputs: List[Output],
        additional_attribute_outputs: Union[List[Output], None],
        feature_outputs: List[Output],
        max_sequence_len: int,
        sample_len: int,
        attribute_noise_dim: int = 10,
        feature_noise_dim: int = 10,
        attribute_num_layers: int = 3,
        attribute_num_units: int = 100,
        feature_num_layers: int = 1,
        feature_num_units: int = 100,
        gradient_penalty_coef: float = 10.0,
        generator_learning_rate: float = 0.001,
        generator_beta1: float = 0.5,
        discriminator_learning_rate: float = 0.001,
        discriminator_beta1: float = 0.5,
        use_attribute_discriminator: bool = False,
        attribute_gradient_penalty_coef: float = 10.0,
        attribute_loss_coef: float = 1.0,
        attribute_discriminator_learning_rate: float = 0.001,
        attribute_discriminator_beta1: float = 0.5,
        forget_bias: bool = False,
    ):
        """
        Args:
            forget_bias: if True, initialize forget gate bias to 1 in LSTM
                layers, otherwise use default pytorch initialization.
                forget_bias=True mimics tf1 LSTMCell behavior.
        """
        if max_sequence_len % sample_len != 0:
            raise RuntimeError(
                f"max_sequence_len={max_sequence_len} must be divisible by sample_len={sample_len}"
            )

        self.EPS = 1e-8
        self.attribute_outputs = attribute_outputs
        self.additional_attribute_outputs = additional_attribute_outputs
        self.feature_outputs = feature_outputs
        self.gradient_penalty_coef = gradient_penalty_coef
        self.generator_learning_rate = generator_learning_rate
        self.generator_beta1 = generator_beta1
        self.discriminator_learning_rate = discriminator_learning_rate
        self.discriminator_beta1 = discriminator_beta1
        self.attribute_gradient_penalty_coef = attribute_gradient_penalty_coef
        self.attribute_loss_coef = attribute_loss_coef
        self.attribute_discriminator_learning_rate = (
            attribute_discriminator_learning_rate
        )
        self.attribute_discriminator_beta1 = attribute_discriminator_beta1

        self.generator = Generator(
            attribute_outputs,
            additional_attribute_outputs,
            feature_outputs,
            max_sequence_len,
            sample_len,
            attribute_noise_dim,
            feature_noise_dim,
            attribute_num_units,
            attribute_num_layers,
            feature_num_units,
            feature_num_layers,
        )

        # TODO: get dims from generator instead of assuming internal details.
        attribute_dim = sum(output.get_dim() for output in attribute_outputs)
        additional_attribute_dim = 0
        if self.additional_attribute_outputs:
            additional_attribute_dim = sum(
                output.get_dim() for output in self.additional_attribute_outputs
            )
        feature_dim = sum(output.get_dim() for output in feature_outputs)

        self.feature_discriminator = Discriminator(
            attribute_dim + additional_attribute_dim + max_sequence_len * feature_dim,
            num_layers=5,
            num_units=200,
        )

        self.attribute_discriminator = None
        if use_attribute_discriminator:
            self.attribute_discriminator = Discriminator(
                attribute_dim + additional_attribute_dim,
                num_layers=5,
                num_units=200,
            )

        self.attribute_noise_func = lambda batch_size: torch.randn(
            batch_size, attribute_noise_dim
        )
        # TODO: add additional attribute noise func
        self.feature_noise_func = lambda batch_size: torch.randn(
            batch_size, max_sequence_len // sample_len, feature_noise_dim
        )

        if forget_bias:

            def init_weights(m):
                if "LSTM" in str(m.__class__):
                    for name, param in m.named_parameters(recurse=False):
                        if "bias_hh" in name:
                            # The LSTM bias param is a concatenation of 4 bias
                            # terms: (b_ii|b_if|b_ig|b_io). We only want to
                            # change the forget gate bias, i.e., b_if. But we
                            # can't change a slice of the tensor, so need to
                            # recreate the initialization for the other parts
                            # and concatenate with the new forget gate bias
                            # initialization.
                            with torch.no_grad():
                                hidden_size = m.hidden_size
                                a = -np.sqrt(1.0 / hidden_size)
                                b = np.sqrt(1.0 / hidden_size)
                                bias_ii = torch.Tensor(hidden_size)
                                bias_ig_io = torch.Tensor(hidden_size * 2)
                                bias_if = torch.Tensor(hidden_size)
                                torch.nn.init.uniform_(bias_ii, a, b)
                                torch.nn.init.uniform_(bias_ig_io, a, b)
                                torch.nn.init.ones_(bias_if)
                                new_param = torch.cat(
                                    [bias_ii, bias_if, bias_ig_io], dim=0
                                )
                                param.copy_(new_param)

            self.generator.apply(init_weights)

    def generate(
        self, batch_size: int = None, attribute_noise=None, feature_noise=None
    ):
        # TODO: what tensor size should be returned for batch_size=1?
        if batch_size is not None:
            attribute_noise = self.attribute_noise_func(batch_size)
            feature_noise = self.feature_noise_func(batch_size)
        else:
            if attribute_noise is None or feature_noise is None:
                raise RuntimeError(
                    "generate() must receive either batch_size or both attribute_noise and feature_noise"
                )

        batch = self.generator(attribute_noise, feature_noise)

        batch = [x.detach().numpy() for x in batch]
        if self.additional_attribute_outputs:
            transformed_attributes, additional_attributes, transformed_features = batch

            attributes = inverse_transform(
                transformed_attributes, self.attribute_outputs, variable_dim_index=1
            )
            features = inverse_transform(
                transformed_features,
                self.feature_outputs,
                variable_dim_index=2,
                additional_attributes=additional_attributes,
                additional_attribute_outputs=self.additional_attribute_outputs,
            )
        else:
            transformed_attributes, transformed_features = batch

            attributes = inverse_transform(
                transformed_attributes, self.attribute_outputs, variable_dim_index=1
            )
            features = inverse_transform(
                transformed_features, self.feature_outputs, variable_dim_index=2
            )

        return attributes, features

    def _discriminate(self, batch):
        inputs = list(batch)
        # Flatten the features
        inputs[-1] = torch.reshape(inputs[-1], (inputs[-1].shape[0], -1))

        input = torch.cat(inputs, dim=1)

        output = self.feature_discriminator(input)
        return output

    def _discriminate_attributes(self, batch):
        if not self.attribute_discriminator:
            raise RuntimeError(
                "discriminate_attributes called with no attribute_discriminator"
            )

        input = torch.cat(batch, dim=1)

        output = self.attribute_discriminator(input)
        return output

    def _get_gradient_penalty(self, generated_batch, real_batch, discriminator_func):
        alpha = torch.rand(generated_batch[0].shape[0])

        interpolated_batch = [
            interpolate(g, r, alpha).requires_grad_(True)
            for g, r in zip(generated_batch, real_batch)
        ]

        interpolated_output = discriminator_func(interpolated_batch)

        gradients = torch.autograd.grad(
            interpolated_output,
            interpolated_batch,
            grad_outputs=torch.ones(interpolated_output.shape),
            retain_graph=True,
            create_graph=True,
        )

        squared_sums = [
            torch.sum(torch.square(g.view(g.size(0), -1))) for g in gradients
        ]

        norm = torch.sqrt(sum(squared_sums) + self.EPS)

        return ((norm - 1.0) ** 2).mean()

    def add_batch_summary(
        self, tb_writer: tensorboard.SummaryWriter, batch, prefix: str, global_step: int
    ):
        """Add histograms and other info to tensorboard for a batch of data, either
        generated or real.
        """
        # Attributes
        attributes = batch[0]
        index = 0
        for output in self.attribute_outputs:
            if "DiscreteOutput" in str(output.__class__):
                probs = attributes[:, index : (index + output.dim)]
                indices = torch.argmax(probs, dim=1)

                tb_writer.add_histogram(
                    prefix + "/attributes/" + output.name, indices, global_step
                )
                tb_writer.add_histogram(
                    prefix + "/attributes/" + output.name + "_probs", probs, global_step
                )

            elif "ContinuousOutput" in str(output.__class__):
                tb_writer.add_histogram(
                    prefix + "/attributes/" + output.name,
                    attributes[:, index],
                    global_step,
                )
            index += output.get_dim()

        if self.additional_attribute_outputs:
            additional_attributes = batch[1]

            index = 0
            for output in self.additional_attribute_outputs:
                if "DiscreteOutput" in str(output.__class__):
                    probs = additional_attributes[:, index : (index + output.dim)]
                    indices = torch.argmax(probs, dim=1)

                    tb_writer.add_histogram(
                        prefix + "/additional_attributes/" + output.name,
                        indices,
                        global_step,
                    )
                    tb_writer.add_histogram(
                        prefix + "/additional_attributes/" + output.name + "_probs",
                        probs,
                        global_step,
                    )
                elif "ContinuousOutput" in str(output.__class__):
                    tb_writer.add_histogram(
                        prefix + "/additional_attributes/" + output.name,
                        additional_attributes[:, index],
                        global_step,
                    )
                index += output.get_dim()

    def train(
        self,
        dataset,
        batch_size: int,
        num_epochs: int,
        discriminator_rounds: int = 1,
        generator_rounds: int = 1,
        tb_writer: tensorboard.SummaryWriter = None,
        log_activations: bool = False,
    ):
        # TODO: performance, cpu/gpu, pinning memory

        # TODO: support arbitrary batch size so we don't need drop_last=True
        loader = torch.utils.data.DataLoader(
            dataset, batch_size, shuffle=True, drop_last=True
        )

        opt_discriminator = torch.optim.Adam(
            self.feature_discriminator.parameters(),
            lr=self.discriminator_learning_rate,
            betas=(self.discriminator_beta1, 0.999),
        )

        opt_attribute_discriminator = None
        if self.attribute_discriminator is not None:
            opt_attribute_discriminator = torch.optim.Adam(
                self.attribute_discriminator.parameters(),
                lr=self.attribute_discriminator_learning_rate,
                betas=(self.attribute_discriminator_beta1, 0.999),
            )

        opt_generator = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.generator_learning_rate,
            betas=(self.generator_beta1, 0.999),
        )

        if tb_writer is not None:
            tb_writer.add_graph(
                self.generator,
                (
                    self.attribute_noise_func(batch_size).detach(),
                    self.feature_noise_func(batch_size).detach(),
                ),
            )

        global_step = 0
        if log_activations and tb_writer is not None:
            # Add forward and backward hooks to log activations and gradients to
            # tensorbord.
            def process_nested(x, prefix, func):
                if isinstance(x, torch.Tensor):
                    func(x, prefix)
                elif isinstance(x, tuple):
                    for index, e in enumerate(x):
                        process_nested(e, "{}_{}".format(prefix, index), func)
                elif x is None:
                    pass
                else:
                    print(f"Unknown input type={type(x)} for process_nested")

            def log_activations_and_gradients(m, prefix):
                def add_activations(t, p):
                    tb_writer.add_histogram(p, t, global_step)

                def add_gradients(t, p):
                    tb_writer.add_histogram(p, t, global_step)
                    tb_writer.add_scalar(p + "_norm", t.norm(2), global_step)

                def forward_hook(model, input, output):
                    process_nested(output, "activations/" + prefix, add_activations)

                def backward_hook(model, grad_input, grad_output):
                    process_nested(grad_output, "gradients/" + prefix, add_gradients)

                m.register_forward_hook(forward_hook)
                m.register_full_backward_hook(backward_hook)

            apply_named(self.generator, "", log_activations_and_gradients)

        # TODO: can we detach for no gradient tracking for lots of stuff like
        # noise creation, generator when computing discriminator loss, etc.,
        # does this actually give us performance benefits?
        for epoch in range(num_epochs):
            print(f"epoch: {epoch}")
            # TODO: assumes dataset is already transformed and will output
            # 2-tuple if no additional attributes or a 3-tuple if additional
            # attributes are present

            if tb_writer is not None:
                log_weights(self.generator, tb_writer, epoch)

            for batch_number, real_batch in tqdm(enumerate(loader)):
                global_step += 1
                attribute_noise = self.attribute_noise_func(batch_size)
                feature_noise = self.feature_noise_func(batch_size)

                generated_batch = self.generator(attribute_noise, feature_noise)

                for index, b in enumerate(generated_batch):
                    if torch.isnan(b).any():
                        print(f"found nans in generated_batch index={index}")

                if tb_writer:
                    self.add_batch_summary(tb_writer, real_batch, "real", global_step)
                    self.add_batch_summary(
                        tb_writer, generated_batch, "generated", global_step
                    )

                for _ in range(discriminator_rounds):
                    # TODO: should new noise be generated
                    opt_discriminator.zero_grad()
                    generated_output = self._discriminate(generated_batch)
                    real_output = self._discriminate(real_batch)

                    loss_generated = torch.mean(generated_output)
                    loss_real = -torch.mean(real_output)
                    loss_gradient_penalty = self._get_gradient_penalty(
                        generated_batch, real_batch, self._discriminate
                    )

                    loss = (
                        loss_generated
                        + loss_real
                        + self.gradient_penalty_coef * loss_gradient_penalty
                    )

                    if tb_writer is not None:
                        tb_writer.add_scalar("loss/discriminator", loss, global_step)
                        tb_writer.add_scalar(
                            "loss/discriminator/generated", loss_generated, global_step
                        )
                        tb_writer.add_scalar(
                            "loss/discriminator/real", loss_real, global_step
                        )
                        tb_writer.add_scalar(
                            "loss/discriminator/gradient_penalty",
                            loss_gradient_penalty,
                            global_step,
                        )

                    loss.backward(retain_graph=True)
                    opt_discriminator.step()

                    if opt_attribute_discriminator is not None:
                        opt_attribute_discriminator.zero_grad()
                        # Exclude features (last element of batches) for
                        # attribute discriminator
                        generated_output = self._discriminate_attributes(
                            generated_batch[:-1]
                        )
                        real_output = self._discriminate_attributes(real_batch[:-1])

                        loss_generated = torch.mean(generated_output)
                        loss_real = -torch.mean(real_output)
                        loss_gradient_penalty = self._get_gradient_penalty(
                            generated_batch[:-1],
                            real_batch[:-1],
                            self._discriminate_attributes,
                        )

                        attribute_loss = (
                            loss_generated
                            + loss_real
                            + self.attribute_gradient_penalty_coef
                            * loss_gradient_penalty
                        )

                        if tb_writer is not None:
                            tb_writer.add_scalar(
                                "loss/attribute_discriminator",
                                attribute_loss,
                                global_step,
                            )
                            tb_writer.add_scalar(
                                "loss/attribute_discriminator/generated",
                                loss_generated,
                                global_step,
                            )
                            tb_writer.add_scalar(
                                "loss/attribute_discriminator/real",
                                loss_real,
                                global_step,
                            )
                            tb_writer.add_scalar(
                                "loss/attribute_discriminator/gradient_penalty",
                                loss_gradient_penalty,
                                global_step,
                            )

                        attribute_loss.backward(retain_graph=True)
                        opt_attribute_discriminator.step()

                for _ in range(generator_rounds):
                    # TODO: should new noise be generated?
                    opt_generator.zero_grad()
                    generated_output = self._discriminate(generated_batch)

                    if self.attribute_discriminator:
                        # Exclude features (last element of batch) before
                        # calling attribute discriminator
                        attribute_generated_output = self._discriminate_attributes(
                            generated_batch[:-1]
                        )

                        loss = -torch.mean(
                            generated_output
                        ) + self.attribute_loss_coef * -torch.mean(
                            attribute_generated_output
                        )
                    else:
                        loss = -torch.mean(generated_output)

                    if tb_writer is not None:
                        tb_writer.add_scalar("loss/generator", loss, global_step)
                    loss.backward()
                    opt_generator.step()

            if tb_writer is not None:
                # Generate sample for tensorboard visualization and debugging.
                # TODO: turn eval mode on and then off again maybe?
                num_samples = 1000
                attributes, features = self.generate(num_samples)

                # TODO: add attribute values to figures
                figs = []
                for z in range(min(5, num_samples)):
                    fig = plt.figure()
                    for index in range(features.shape[2]):
                        ax = fig.add_subplot(1, features.shape[2], index + 1)
                        ax.plot(features[z, :, index])

                    figs.append(fig)

                tb_writer.add_figure("generated", figs, global_step=epoch, close=True)

                figs = []
                for index in range(features.shape[2]):
                    acv = autocovariance(features[:, :, index])
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(acv)
                    figs.append(fig)
                tb_writer.add_figure(
                    "generated/autocovariance", figs, global_step=epoch, close=True
                )

        if tb_writer is not None:
            log_weights(self.generator, tb_writer, num_epochs)

        if tb_writer is not None:
            tb_writer.flush()
