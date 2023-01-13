from src.models.model import Transformer

model = Transformer()
devices = ["cpu", "CUDA"]

def test_device():
    assert model.device in devices, "device is not recognized"

num_labels = 5

def test_num_labels():
    assert model.num_labels == num_labels, "num_labels is not recognized"


def test_size():
    assert model.__sizeof__() == 32, "size of model misshapen"
    assert model.parameters().__sizeof__() == 96, "wrong number of parameters"

def test_input_embeddings():
    assert model.get_input_embeddings().embedding_dim == 768, "input is misshapen"
