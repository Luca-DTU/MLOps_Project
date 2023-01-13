model = ()
devices = ["cpu", "CUDA"]
for dev in devices:
    assert model.device == dev, "device is not recognized"
num_labels = 5
assert model.num_labels == num_labels, "model output does not match expected"
assert model.__sizeof__() == 32, "size of model misshapen"
assert model.parameters().__sizeof__() == 96, "wrong number of parameters"
assert model.get_input_embeddings().embedding_dim == 0, "input is misshapen"
