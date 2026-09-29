"""Generate the bundled sample batches used for end-to-end verification."""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path


def cube_data():
    # One quad per face: 4 vertices, 2 triangles. Duplicated seam vertices give
    # explicit normals and UVs while keeping the geometry trivially checkable.
    face_specs = [
        ((1, 0, 0), (0, 0, 0), (0, 1, 0)),
        ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        ((0, 1, 0), (0, 0, 1), (0, 0, 0)),
        ((0, -1, 0), (0, 0, 0), (0, 0, 1)),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        ((0, 0, -1), (0, 0, 0), (0, 1, 0)),
    ]
    positions: list[list[float]] = []
    normals: list[list[float]] = []
    uvs: list[list[float]] = []
    triangles: list[list[int]] = []
    for normal, u_axis, v_axis in face_specs:
        start = len(positions)
        corners = ((0, 0), (1, 0), (1, 1), (0, 1))
        for u_value, v_value in corners:
            position = [
                (0.5 if normal[axis] else 0.0)
                + (u_value - 0.5) * u_axis[axis]
                + (v_value - 0.5) * v_axis[axis]
                for axis in range(3)
            ]
            positions.append(position)
            normals.append(list(normal))
            uvs.append([float(u_value), float(v_value)])
        triangles.extend([
            [start, start + 1, start + 2],
            [start, start + 2, start + 3],
        ])
    return positions, normals, uvs, triangles


def pack_vertices(positions, normals, uvs):
    chunks = []
    for values in positions:
        chunks.append(struct.pack("<3f", *values))
    for values in normals:
        chunks.append(struct.pack("<3f", *values))
    for values in uvs:
        chunks.append(struct.pack("<2f", *values))
    return b"".join(chunks)


def png_checkerboard() -> bytes:
    width = height = 8
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            value = 220 if (x // 2 + y // 2) % 2 else 40
            row.extend((value, max(0, value - 60), 255 - value))
        rows.append(bytes(row))
    raw = b"".join(rows)
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def create_valid_batch(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    positions, normals, uvs, triangles = cube_data()
    vertex_bytes = pack_vertices(positions, normals, uvs)
    indices = [index for triangle in triangles for index in triangle]
    index_bytes = b"".join(struct.pack("<H", index) for index in indices)
    buffer_bytes = vertex_bytes + index_bytes
    (directory / "rig.bin").write_bytes(buffer_bytes)
    (directory / "colors.png").write_bytes(png_checkerboard())

    position_offset = 0
    normal_offset = len(positions) * 12
    uv_offset = normal_offset + len(normals) * 12
    indices_offset = uv_offset + len(uvs) * 8
    gltf = {
        "asset": {"version": "2.0", "generator": "r3d sample"},
        "buffers": [{"uri": "rig.bin", "byteLength": len(buffer_bytes)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": position_offset, "byteLength": len(positions) * 12},
            {"buffer": 0, "byteOffset": normal_offset, "byteLength": len(normals) * 12},
            {"buffer": 0, "byteOffset": uv_offset, "byteLength": len(uvs) * 8},
            {"buffer": 0, "byteOffset": indices_offset, "byteLength": len(index_bytes)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": len(uvs), "type": "VEC2"},
            {"bufferView": 3, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
        ],
        "images": [{"uri": "colors.png", "mimeType": "image/png"}],
        "textures": [{"source": 0}],
        "materials": [
            {
                "name": "blue-checker",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.1, 0.35, 0.9, 1.0],
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.75,
                },
            },
            {
                "name": "red-solid",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.9, 0.12, 0.08, 1.0],
                    "metallicFactor": 0.2,
                    "roughnessFactor": 0.45,
                },
            },
        ],
        "meshes": [{
            "name": "shared-cube",
            "primitives": [
                {"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0},
                {"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 1},
            ],
        }],
        "nodes": [
            {"name": "rig-root", "translation": [0, 1, 0], "children": [1, 3]},
            {"name": "left-box", "mesh": 0, "translation": [-1.5, 0, 0], "children": [2]},
            {"name": "left-cap", "mesh": 0, "scale": [0.5, 0.5, 0.5], "translation": [0, 2, 0]},
            {"name": "right-box", "mesh": 0, "translation": [1.5, 0, 0], "rotation": [0, 0, 0.3826834324, 0.9238795325]},
        ],
        "scenes": [{"name": "nested-cubes", "nodes": [0]}],
        "scene": 0,
    }
    path = directory / "rig.gltf"
    path.write_text(json.dumps(gltf, indent=2), "utf-8")
    return path


def create_broken_batch(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    missing = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "does-not-exist.bin", "byteLength": 12}],
        "bufferViews": [{"buffer": 0, "byteLength": 12}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
    }
    (directory / "missing-buffer.gltf").write_text(json.dumps(missing), "utf-8")
    escaping = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "../outside.bin", "byteLength": 0}],
        "scenes": [],
    }
    (directory / "escaping-uri.gltf").write_text(json.dumps(escaping), "utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="samples")
    args = parser.parse_args()
    root = Path(args.output)
    valid_path = create_valid_batch(root / "sample_batch")
    create_broken_batch(root / "broken_batch")
    print(f"created {valid_path}")
    print(f"created {root / 'broken_batch' / 'missing-buffer.gltf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
