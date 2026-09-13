import onnx

model = onnx.load("deploy/unitree_policy.onnx")
for initializer in model.graph.initializer:
    print(f"{initializer.name}: shape={list(initializer.dims)}")