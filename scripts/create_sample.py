from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def write_sample(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    positions = [
        (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
        (0.0, 1.0, 0.0),
    ]
    cube_indices = [
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2,
        2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0,
    ]
    pyramid_indices = [0, 1, 8, 1, 5, 8, 5, 4, 8, 4, 0, 8, 0, 4, 5, 0, 5, 1]
    position_bytes = struct.pack("<" + "f" * (len(positions) * 3), *(value for p in positions for value in p))
    index_offset = len(position_bytes)
    indices = cube_indices + pyramid_indices
    index_bytes = struct.pack("<" + "H" * len(indices), *indices)
    blob = position_bytes + index_bytes
    (destination / "robot.bin").write_bytes(blob)

    document = {
        "asset": {"version": "2.0", "generator": "resource3d sample"},
        "buffers": [{"uri": "robot.bin", "byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes)},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": len(index_bytes)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "byteOffset": 0, "count": len(cube_indices), "type": "SCALAR"},
            {"bufferView": 1, "componentType": 5123, "byteOffset": len(cube_indices) * 2, "count": len(pyramid_indices), "type": "SCALAR"},
        ],
        "materials": [
            {"name": "blue-metal", "pbrMetallicRoughness": {"baseColorFactor": [0.1, 0.25, 0.8, 1], "metallicFactor": 0.8, "roughnessFactor": 0.35}},
            {"name": "orange-paint", "pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.45, 0.08, 1], "metallicFactor": 0.1, "roughnessFactor": 0.7}},
        ],
        "meshes": [
            {"name": "body", "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]},
            {"name": "sensor-head", "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 1}]},
            {"name": "antenna", "primitives": [{"attributes": {"POSITION": 0}, "indices": 2, "material": 1}]},
        ],
        "nodes": [
            {"name": "world", "rotation": [0, math.sin(math.pi / 8), 0, math.cos(math.pi / 8)], "children": [1, 3]},
            {"name": "body-node", "mesh": 0, "translation": [0, 0, 0], "children": [2]},
            {"name": "head-node", "mesh": 1, "translation": [1.2, 0.8, 0], "scale": [0.5, 0.5, 0.5]},
            {"name": "antenna-node", "mesh": 2, "translation": [-1.1, 0.6, 0.2], "scale": [0.4, 0.7, 0.4]},
        ],
        "scenes": [{"name": "sample-robot", "nodes": [0]}],
        "scene": 0,
    }
    manifest = destination / "robot.gltf"
    manifest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", nargs="?", default="samples/robot")
    args = parser.parse_args()
    print(write_sample(Path(args.destination)))


if __name__ == "__main__":
    main()
