import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from doppelganger import (ContinuousOutput, DGTorch, DiscreteOutput,
                          Normalization, Output, OutputType, prepare_data)


@pytest.fixture
def dg_model() -> DGTorch:
    attribute_outputs = [
        ContinuousOutput(
            name="a",
            normalization=Normalization.ZERO_ONE,
            global_min=0,
            global_max=1,
            is_feature_normalized=False,
            is_example_normalized=False,
        ),
        DiscreteOutput(name="b", dim=3),
    ]
    feature_outputs = [
        DiscreteOutput(name="c", dim=4),
        ContinuousOutput(
            name="d",
            normalization=Normalization.ZERO_ONE,
            global_min=0,
            global_max=1,
            is_feature_normalized=False,
            is_example_normalized=False,
        ),
    ]
    dg = DGTorch(attribute_outputs, [], feature_outputs, 20, 5)
    return dg


@pytest.fixture
def attribute_data():
    n = 100
    attributes = np.concatenate(
        (
            np.random.rand(n, 1),
            np.random.randint(0, 3, size=(n, 1)),
        ),
        axis=1,
    )
    return (attributes, [OutputType.CONTINUOUS, OutputType.DISCRETE])


@pytest.fixture
def feature_data():
    n = 100
    features = np.concatenate(
        (
            np.random.randint(0, 4, size=(n, 20, 1)),
            np.random.rand(n, 20, 1),
        ),
        axis=2,
    )
    return (features, [OutputType.DISCRETE, OutputType.CONTINUOUS])


def test_generate(dg_model: DGTorch):
    attributes, features = dg_model.generate(8)

    assert attributes.shape == (8, 2)
    assert features.shape == (8, 20, 2)


def test_train(attribute_data, feature_data):
    attributes, attribute_types = attribute_data
    features, feature_types = feature_data

    dg_data = prepare_data(
        attributes,
        attribute_types,
        features,
        feature_types,
        is_feature_normalized=False,
        is_example_normalized=False,
    )

    dg = DGTorch(
        attribute_outputs=dg_data.attribute_outputs,
        additional_attribute_outputs=None,
        feature_outputs=dg_data.feature_outputs,
        max_sequence_len=20,
        sample_len=5,
    )

    dataset = torch.utils.data.TensorDataset(
        torch.Tensor(dg_data.attributes),
        torch.Tensor(dg_data.features),
    )
    dg.train(dataset, batch_size=10, num_epochs=2)


def test_train_with_attribute_discriminator(attribute_data, feature_data):
    attributes, attribute_types = attribute_data
    features, feature_types = feature_data

    dg_data = prepare_data(
        attributes,
        attribute_types,
        features,
        feature_types,
        is_feature_normalized=False,
        is_example_normalized=False,
    )

    dg = DGTorch(
        attribute_outputs=dg_data.attribute_outputs,
        additional_attribute_outputs=None,
        feature_outputs=dg_data.feature_outputs,
        max_sequence_len=20,
        sample_len=5,
        use_attribute_discriminator=True,
    )

    dataset = torch.utils.data.TensorDataset(
        torch.Tensor(dg_data.attributes),
        torch.Tensor(dg_data.features),
    )
    dg.train(dataset, batch_size=10, num_epochs=2)


def test_train_with_additional_attributes(attribute_data, feature_data):
    attributes, attribute_types = attribute_data
    features, feature_types = feature_data

    dg_data = prepare_data(
        attributes,
        attribute_types,
        features,
        feature_types,
        is_feature_normalized=False,
        is_example_normalized=True,
    )

    dg = DGTorch(
        attribute_outputs=dg_data.attribute_outputs,
        additional_attribute_outputs=dg_data.additional_attribute_outputs,
        feature_outputs=dg_data.feature_outputs,
        max_sequence_len=20,
        sample_len=5,
    )

    dataset = torch.utils.data.TensorDataset(
        torch.Tensor(dg_data.attributes),
        torch.Tensor(dg_data.additional_attributes),
        torch.Tensor(dg_data.features),
    )
    dg.train(dataset, batch_size=10, num_epochs=2)


def test_train_with_additional_attributes_and_discriminator(
    attribute_data, feature_data
):
    attributes, attribute_types = attribute_data
    features, feature_types = feature_data

    dg_data = prepare_data(
        attributes,
        attribute_types,
        features,
        feature_types,
        is_feature_normalized=False,
        is_example_normalized=True,
    )

    dg = DGTorch(
        attribute_outputs=dg_data.attribute_outputs,
        additional_attribute_outputs=dg_data.additional_attribute_outputs,
        feature_outputs=dg_data.feature_outputs,
        max_sequence_len=20,
        sample_len=5,
        use_attribute_discriminator=True,
    )

    dataset = torch.utils.data.TensorDataset(
        torch.Tensor(dg_data.attributes),
        torch.Tensor(dg_data.additional_attributes),
        torch.Tensor(dg_data.features),
    )
    dg.train(dataset, batch_size=10, num_epochs=2)


def test_output():
    o1 = Output(name="foo")
    assert o1.name == "foo"
    assert o1.get_dim() == 1

    o2 = DiscreteOutput(name="foo", dim=4)
    assert o2.name == "foo"
    assert o2.dim == 4
    assert o2.get_dim() == 4

    o3 = ContinuousOutput(
        name="foo",
        normalization=Normalization.ZERO_ONE,
        global_min=0.0,
        global_max=1.0,
        is_feature_normalized=False,
        is_example_normalized=False,
    )
    assert o3.get_dim() == 1

    with pytest.raises(TypeError):
        DiscreteOutput(name="bad")

    with pytest.raises(TypeError):
        ContinuousOutput(
            name="bad",
            normalization=Normalization.ZERO_ONE,
            global_min=0,
            global_max=1,
            dim=5,
            is_feature_normalized=False,
            is_example_normalized=False,
        )


def test_prepare_data():
    original_attributes = np.concatenate(
        (
            np.random.rand(100, 1),
            np.random.randint(0, 3, size=(100, 1)),
        ),
        axis=1,
    )
    original_features = np.concatenate(
        (
            np.random.rand(100, 20, 1),
            np.random.randint(0, 2, size=(100, 20, 1)),
        ),
        axis=2,
    )

    attribute_types = [OutputType.CONTINUOUS, OutputType.DISCRETE]
    feature_types = [OutputType.CONTINUOUS, OutputType.DISCRETE]

    dg_data = prepare_data(
        original_attributes, attribute_types, original_features, feature_types
    )

    assert dg_data.attributes.shape == (100, 4)
    assert dg_data.additional_attributes.shape == (100, 2)
    assert dg_data.features.shape == (100, 20, 3)
