import onnx

input_path = "nanodet-plus-m_416.onnx"
output_path = "nanodet-plus-m_416-sub.onnx"
input_names = ["data"]
output_names = [
    "1445",
    "1470",
    "1495",
    "1520",
]

onnx.utils.extract_model(input_path, output_path, input_names, output_names)
