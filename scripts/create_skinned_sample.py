from __future__ import annotations

import argparse
import base64
import json
import math
import struct
from pathlib import Path


def write_skinned_sample(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    # Two triangles forming a bar along X: [0,0]-[2,0] in the XY plane.
    positions = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0),
        (1.0, 0.0, 0.0), (2.0, 1.0, 0.0), (1.0, 1.0, 0.0),
    ]
    unique_positions = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (2.0, 1.0, 0.0),
    ]
    indices = [0, 1, 4, 0, 4, 3, 1, 2, 5, 1, 5, 4]
    # Joint 0 for left column, joint 1 for right column, blended at x=1.
    joint_vectors = [
        (0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0),
        (0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0),
    ]
    weight_vectors = [
        (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
    ]

    def pack_floats(vectors):
        flat = [value for vector in vectors for value in vector]
        return struct.pack("<" + "f" * len(flat), *flat)

    position_bytes = pack_floats(unique_positions)
    index_bytes = struct.pack("<" + "H" * len(indices), *indices)
    joint_bytes = struct.pack("<" + "B" * (len(joint_vectors) * 4), *(v for jv in joint_vectors for v in jv))
    weight_bytes = pack_floats(weight_vectors)
    offset_0 = 0
    offset_1 = len(position_bytes)
    offset_2 = offset_1 + len(index_bytes)
    offset_3 = offset_2 + len(joint_bytes)
    blob = position_bytes + index_bytes + joint_bytes + weight_bytes
    (destination / "character.bin").write_bytes(blob)

    # Two distinct images. The same color image is referenced by two
    # materials and also reused in a second texture slot of the data map.
    color_png = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + b"COLOR-SURFACE"
    data_png = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + b"METAL-ROUGH-DATA"
    (destination / "paint.png").write_bytes(color_png)
    data_uri = "data:image/png;base64," + base64.b64encode(data_png).decode("ascii")

    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    # Joint 1 is authored at x=1, so its inverse bind matrix translates back.
    inverse_joint_1 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 1]
    ibm_bytes = struct.pack("<" + "f" * 32, *(identity + inverse_joint_1))

    document = {
        "asset": {"version": "2.0", "generator": "resource3d skinned sample"},
        "buffers": [
            {"uri": "character.bin", "byteLength": len(blob)},
            {"uri": "data:application/octet-stream;base64," + base64.b64encode(ibm_bytes).decode("ascii"), "byteLength": len(ibm_bytes)},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offset_0, "byteLength": len(position_bytes)},
            {"buffer": 0, "byteOffset": offset_1, "byteLength": len(index_bytes)},
            {"buffer": 0, "byteOffset": offset_2, "byteLength": len(joint_bytes)},
            {"buffer": 0, "byteOffset": offset_3, "byteLength": len(weight_bytes)},
            {"buffer": 1, "byteOffset": 0, "byteLength": len(ibm_bytes)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(unique_positions), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "byteOffset": 0, "count": len(indices), "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5121, "count": len(joint_vectors), "type": "VEC4"},
            {"bufferView": 3, "componentType": 5126, "count": len(weight_vectors), "type": "VEC4"},
            {"bufferView": 4, "componentType": 5126, "count": 2, "type": "MAT4"},
        ],
        "images": [
            {"name": "paint", "uri": "paint.png"},
            {"name": "metal-rough", "uri": data_uri},
        ],
        "textures": [
            {"name": "paint-srgb", "source": 0},
            {"name": "metal-rough-linear", "source": 1},
        ],
        "materials": [
            {
                "name": "armor",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1, 1, 1, 1],
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 1},
                },
            },
            {
                "name": "trim",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.9, 0.9, 0.9, 1],
                    "baseColorTexture": {"index": 0, "texCoord": 0},
                },
            },
        ],
        "meshes": [
            {
                "name": "skinned-bar",
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
            {
                "name": "bar-skin",
                "inverseBindMatrices": 4,
                "joints": [1, 2],
                "skeleton": 1,
            }
        ],
        "nodes": [
            {
                "name": "rig",
                "rotation": [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)],
                "children": [1, 3],
            },
            {"name": "joint-hip", "translation": [0, 0, 0], "children": [2]},
            {"name": "joint-tip", "translation": [1, 0, 0]},
            {
                "name": "bar-node",
                "mesh": 0,
                "skin": 0,
                "translation": [0, -2, 0],
            },
        ],
        "scenes": [{"name": "skinned-character", "nodes": [0]}],
        "scene": 0,
    }
    manifest = destination / "character.gltf"
    manifest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", nargs="?", default="samples/character")
    args = parser.parse_args()
    print(write_skinned_sample(Path(args.destination)))


if __name__ == "__main__":
    main()
