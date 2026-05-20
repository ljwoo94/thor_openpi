#!/usr/bin/env python3
"""Inspect ONNX precision and QDQ coverage for OpenPI TensorRT exports."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import onnx
from onnx import TensorProto


FORBIDDEN_QDQ_PATTERNS = (
    "vision_tower",
    "embeddings",
    "patch_embedding",
    "position_embedding",
    "layer_norm",
    "layernorm",
    "input_layernorm",
    "post_attention_layernorm",
    "post_layernorm",
    "model.norm",
)


def _dtype_name(elem_type: int) -> str:
    try:
        return TensorProto.DataType.Name(elem_type)
    except ValueError:
        return f"UNKNOWN({elem_type})"


def _value_info_dtype(value_info) -> str:
    tensor_type = value_info.type.tensor_type
    if tensor_type.elem_type == TensorProto.UNDEFINED:
        return "UNKNOWN"
    return _dtype_name(tensor_type.elem_type)


def inspect_model(path: Path) -> None:
    model = onnx.load(path, load_external_data=False)
    graph = model.graph

    op_counts = Counter(node.op_type for node in graph.node)
    initializer_dtypes = Counter(_dtype_name(init.data_type) for init in graph.initializer)
    input_dtypes = {value.name: _value_info_dtype(value) for value in graph.input}
    output_dtypes = {value.name: _value_info_dtype(value) for value in graph.output}

    producer_by_output = {}
    consumers_by_input = {}
    for node in graph.node:
        for output in node.output:
            producer_by_output[output] = node
        for input_name in node.input:
            consumers_by_input.setdefault(input_name, []).append(node)

    qdq_ops = {"QuantizeLinear", "DequantizeLinear"}
    qdq_nodes = [node for node in graph.node if node.op_type in qdq_ops]
    matmul_nodes = [node for node in graph.node if node.op_type in {"MatMul", "Gemm"}]
    cast_nodes = [node for node in graph.node if node.op_type == "Cast"]

    qdq_adjacent_matmuls = []
    for node in matmul_nodes:
        has_qdq_input = any(
            input_name in producer_by_output and producer_by_output[input_name].op_type in qdq_ops
            for input_name in node.input
        )
        has_qdq_output = any(
            consumer.op_type in qdq_ops
            for output_name in node.output
            for consumer in consumers_by_input.get(output_name, [])
        )
        if has_qdq_input or has_qdq_output:
            qdq_adjacent_matmuls.append(node)

    forbidden_qdq = [
        node
        for node in qdq_nodes
        if any(pattern in node.name.lower() for pattern in FORBIDDEN_QDQ_PATTERNS)
    ]

    print(f"ONNX: {path}")
    print(f"IR version: {model.ir_version}")
    opsets = ", ".join(f"{op.domain or 'ai.onnx'}:{op.version}" for op in model.opset_import)
    print(f"Opset imports: {opsets}")
    print("")

    print("Inputs:")
    for name, dtype in input_dtypes.items():
        print(f"  {name}: {dtype}")
    print("Outputs:")
    for name, dtype in output_dtypes.items():
        print(f"  {name}: {dtype}")
    print("")

    print("Initializer dtypes:")
    for dtype, count in sorted(initializer_dtypes.items()):
        print(f"  {dtype}: {count}")
    print("")

    print("Key op counts:")
    for op_type in ("QuantizeLinear", "DequantizeLinear", "Cast", "MatMul", "Gemm", "Softmax", "LayerNormalization"):
        print(f"  {op_type}: {op_counts.get(op_type, 0)}")
    print("")

    print(f"QDQ-adjacent MatMul/Gemm nodes: {len(qdq_adjacent_matmuls)}")
    for node in qdq_adjacent_matmuls[:30]:
        print(f"  {node.op_type}: {node.name or '<unnamed>'}")
    if len(qdq_adjacent_matmuls) > 30:
        print(f"  ... {len(qdq_adjacent_matmuls) - 30} more")
    print("")

    print(f"Cast nodes: {len(cast_nodes)}")
    for node in cast_nodes[:30]:
        print(f"  Cast: {node.name or '<unnamed>'}")
    if len(cast_nodes) > 30:
        print(f"  ... {len(cast_nodes) - 30} more")
    print("")

    if forbidden_qdq:
        print("WARNING: QDQ nodes matched sensitive-name patterns:")
        for node in forbidden_qdq[:50]:
            print(f"  {node.op_type}: {node.name or '<unnamed>'}")
        if len(forbidden_qdq) > 50:
            print(f"  ... {len(forbidden_qdq) - 50} more")
    else:
        print("No QDQ nodes matched sensitive-name patterns.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_path", type=Path, help="Path to ONNX model")
    args = parser.parse_args()
    inspect_model(args.onnx_path)


if __name__ == "__main__":
    main()
