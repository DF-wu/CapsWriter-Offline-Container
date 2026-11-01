"""Promote an existing FP16 ONNX graph; this cannot restore rounded weights.

Run inside the disposable runtime, with onnx==1.16.2 installed:
  python promote_encoder.py INPUT.onnx OUTPUT.onnx
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def promote_tensor(tensor):
    if tensor.data_type == TensorProto.FLOAT16:
        replacement = numpy_helper.from_array(
            numpy_helper.to_array(tensor).astype(np.float32), tensor.name
        )
        tensor.CopyFrom(replacement)


def promote_graph(graph):
    for value in [*graph.input, *graph.output, *graph.value_info]:
        if value.type.tensor_type.elem_type == TensorProto.FLOAT16:
            value.type.tensor_type.elem_type = TensorProto.FLOAT
    for tensor in graph.initializer:
        promote_tensor(tensor)
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                promote_tensor(attr.t)
            elif attr.type == onnx.AttributeProto.TENSORS:
                for tensor in attr.tensors:
                    promote_tensor(tensor)
            elif attr.type == onnx.AttributeProto.GRAPH:
                promote_graph(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for child in attr.graphs:
                    promote_graph(child)
            if node.op_type == 'Cast' and attr.name == 'to' and attr.i == TensorProto.FLOAT16:
                attr.i = TensorProto.FLOAT
    # The published backend has Casts after their consumers. ORT accepts it,
    # but ONNX checker requires a topological ordering.
    known = {value.name for value in graph.input} | {t.name for t in graph.initializer}
    pending, ordered = list(graph.node), []
    while pending:
        remaining, ready = [], []
        for node in pending:
            if all(not name or name in known for name in node.input):
                ready.append(node)
            else:
                remaining.append(node)
        if not ready:
            raise ValueError('Unresolved graph dependencies')
        ordered.extend(ready)
        for node in ready:
            known.update(node.output)
        pending = remaining
    del graph.node[:]
    graph.node.extend(ordered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    model = onnx.load(args.source)
    promote_graph(model.graph)
    onnx.checker.check_model(model)
    onnx.save(model, args.output)


if __name__ == '__main__':
    main()
