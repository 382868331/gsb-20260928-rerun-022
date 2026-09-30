from __future__ import annotations

import argparse
import json
import math
import struct
import zlib
from pathlib import Path


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_png(path: Path, gray: int) -> None:
    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    scanline = b"\x00" + bytes([gray, gray, gray])
    image = png_chunk(b"IHDR", header) + png_chunk(b"IDAT", zlib.compress(scanline)) + png_chunk(b"IEND", b"")
    path.write_bytes(signature + image)


def write_sample(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    write_png(destination / "shared.png", 200)
    write_png(destination / "packed.png", 40)

    positions = [
        (-0.5, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (-0.5, 1.0, 0.0),
        (0.5, 1.0, 0.0),
    ]
    indices = [0, 1, 2, 2, 1, 3]
    joints = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]
    weights = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    inverse_bind = [
        [1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, -0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    position_bytes = struct.pack("<" + "f" * 12, *(value for vertex in positions for value in vertex))
    index_bytes = struct.pack("<HHHHHH", *indices)
    joint_bytes = b"".join(struct.pack("<BBBB", *quad) for quad in joints)
    weight_bytes = struct.pack("<" + "f" * 16, *(value for quad in weights for value in quad))
    ibm_bytes = struct.pack("<" + "f" * 32, *(value for matrix in inverse_bind for value in matrix))
    offset_indices = len(position_bytes)
    offset_joints = offset_indices + len(index_bytes)
    offset_weights = offset_joints + len(joint_bytes)
    offset_ibm = offset_weights + len(weight_bytes)
    blob = position_bytes + index_bytes + joint_bytes + weight_bytes + ibm_bytes
    (destination / "rig.bin").write_bytes(blob)

    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "rig.bin", "byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes)},
            {"buffer": 0, "byteOffset": offset_indices, "byteLength": len(index_bytes)},
            {"buffer": 0, "byteOffset": offset_joints, "byteLength": len(joint_bytes)},
            {"buffer": 0, "byteOffset": offset_weights, "byteLength": len(weight_bytes)},
            {"buffer": 0, "byteOffset": offset_ibm, "byteLength": len(ibm_bytes)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5121, "count": 4, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5126, "count": 4, "type": "VEC4"},
            {"bufferView": 4, "componentType": 5126, "count": 2, "type": "MAT4"},
        ],
        "images": [
            {"uri": "shared.png"},
            {"uri": "packed.png"},
        ],
        "textures": [
            {"source": 0, "name": "shared"},
            {"source": 1, "name": "packed"},
        ],
        "materials": [
            {
                "name": "red-cloth",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 1},
                },
            },
            {
                "name": "blue-cloth",
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                "normalTexture": {"index": 1},
            },
        ],
        "meshes": [
            {
                "name": "skinned-panel",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "JOINTS_0": 2,
                            "WEIGHTS_0": 3,
                        },
                        "indices": 1,
                        "material": 0,
                    }
                ],
            }
        ],
        "skins": [
            {"name": "panel-rig", "inverseBindMatrices": 4, "joints": [2, 3]},
        ],
        "nodes": [
            {"name": "rig-root", "rotation": [0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)], "children": [1, 2, 3]},
            {"name": "panel-node", "mesh": 0, "skin": 0},
            {"name": "joint-left", "translation": [-0.5, 0.0, 0.0]},
            {"name": "joint-right", "translation": [0.5, 0.0, 0.0]},
        ],
        "scenes": [{"name": "rig-scene", "nodes": [0]}],
        "scene": 0,
    }
    manifest = destination / "rig.gltf"
    manifest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", nargs="?", default="samples/rig")
    args = parser.parse_args()
    print(write_sample(Path(args.destination)))


if __name__ == "__main__":
    main()
