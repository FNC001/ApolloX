"""PDM embedding uses real tensors and keeps the existing MLP parameter names."""

import pytest
import torch

from cdvae.pl_modules.conditioning import MultiEmbedding, VectorEmbedding, ZGivenC


def vector_config():
    return {
        "_target_": "cdvae.pl_modules.conditioning.VectorEmbedding",
        "prop_name": "element_values",
        "n_in": 3,
        "hidden_dim": 4,
        "fc_num_layers": 1,
        "n_out": 2,
    }


def test_multicondition_vector_changes_conditional_latent_and_receives_gradients():
    embedding = MultiEmbedding(
        ["pressure", "element_values"],
        {
            "pressure": {
                "_target_": "cdvae.pl_modules.conditioning.ScalarEmbedding",
                "prop_name": "pressure",
                "no_expansion": True,
                "no_mlp": True,
                "n_out": 1,
            },
            "element_values": vector_config(),
        },
    )
    for parameter in embedding.parameters():
        torch.nn.init.constant_(parameter, 0.25)
    combine = ZGivenC(mode="concat", no_mlp=True)
    z = torch.zeros(2, 2)
    first = torch.ones(2, 3, requires_grad=True)
    second = torch.full((2, 3), 3.0)
    c1 = embedding({"pressure": torch.zeros(2, 1), "element_values": first})
    c2 = embedding({"pressure": torch.zeros(2, 1), "element_values": second})
    latent1, latent2 = combine(z, c1), combine(z, c2)
    assert latent1.shape == (2, 5)
    torch.testing.assert_close(latent1[:, :3], latent2[:, :3])
    assert not torch.equal(latent1[:, 3:], latent2[:, 3:])
    latent1.sum().backward()
    assert torch.isfinite(first.grad).all() and (first.grad != 0).all()
    assert any(parameter.grad is not None for parameter in embedding.parameters())


@pytest.mark.parametrize(
    "values, error",
    [
        (torch.ones(3), "shape"),
        (torch.ones(2, 2), "shape"),
        (torch.ones(2, 1, 3), "shape"),
        (torch.tensor([[1.0, float("nan"), 2.0]]), "finite"),
        (torch.tensor([[1.0, float("inf"), 2.0]]), "finite"),
    ],
)
def test_vector_embedding_rejects_invalid_tensor(values, error):
    embedding = VectorEmbedding("element_values", 3, 4, 1, 2)
    with pytest.raises(ValueError, match=error):
        embedding(values)


def test_vector_embedding_rejects_mapping_and_preserves_checkpoint_keys():
    embedding = VectorEmbedding("element_values", 3, 4, 1, 2)
    with pytest.raises(TypeError, match="must be a tensor"):
        embedding({"element_values": torch.ones(1, 3)})
    assert set(embedding.state_dict()) == {
        "mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias"
    }


@pytest.mark.parametrize("n_in", [None, 0, -1, 2.5, True])
def test_vector_embedding_requires_explicit_positive_dimension(n_in):
    with pytest.raises(ValueError, match="positive integer"):
        VectorEmbedding("element_values", n_in, 4, 1, 2)
