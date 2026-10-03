from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def make_png(width: int, height: int, seed: int) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = bytes([seed % 256]) * (width * 3)
    raw = b"".join(b"\x00" + row for _ in range(height))
    return (
        PNG_SIGNATURE
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def write_batch(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    blob = struct.pack("<9f3H", *(value for vector in positions for value in vector), 0, 1, 2)
    (destination / "mesh.bin").write_bytes(blob)

    seed = 10
    shared_payload = make_png(4, 4, seed=seed)
    # color[0] and normal[0] point at the same image file/bytes; color[1] and
    # normal[1] get independent payloads.
    files = {
        "shared-0-0.png": shared_payload,
        "color-1-0.png": make_png(4, 4, seed=seed + 1),
        "normal-1-0.png": make_png(4, 4, seed=seed + 3),
    }
    for name, payload in files.items():
        (destination / name).write_bytes(payload)
    images = [{"name": Path(name).stem, "uri": name} for name in files]
    image_index = {image["uri"]: index for index, image in enumerate(images)}
    # One glTF texture per usage/position even when two share one image.
    texture_specs = [
        ("color", 0, "shared-0-0.png"),
        ("color", 1, "color-1-0.png"),
        ("normal", 0, "shared-0-0.png"),
        ("normal", 1, "normal-1-0.png"),
    ]
    textures = [
        {
            "name": f"{usage}-{column}",
            "source": image_index[file_name],
            "extras": {"tile": {"position": [column, 0], "layout": [2, 1]}},
        }
        for usage, column, file_name in texture_specs
    ]
    texture_key = {(usage, column): index for index, (usage, column, _) in enumerate(texture_specs)}
    color_files = ["shared-0-0.png", "color-1-0.png"]
    normal_files = ["shared-0-0.png", "normal-1-0.png"]

    def tile_entries(usage_files: list[str]) -> list[dict]:
        entries = []
        for column, name in enumerate(usage_files):
            usage = "color" if usage_files is color_files else "normal"
            texture_index = texture_key[(usage, column)]
            entries.append({
                "index": texture_index,
                "extras": {"tile": {"position": [column, 0], "layout": [2, 1],
                                   "edges": [[], ["seam"], [], []] if column == 0
                                   else [["seam"], [], [], []]}},
            })
        return entries

    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "mesh.bin", "byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "images": images,
        "textures": textures,
        "materials": [{
            "name": "tiled-mat",
            "extras": {
                "tiles": {
                    "color": tile_entries(color_files),
                    "normal": tile_entries(normal_files),
                }
            },
        }],
        "meshes": [{
            "name": "tri",
            "primitives": [
                {"attributes": {"POSITION": 0}, "indices": 1, "material": 0}
            ],
        }],
        "nodes": [
            {"name": "node-a", "mesh": 0},
            {"name": "node-b", "mesh": 0},
        ],
        "scenes": [{"name": "scene", "nodes": [0, 1]}],
        "scene": 0,
    }
    manifest = destination / "model.gltf"
    manifest.write_text(json.dumps(document, indent=2) + "\n")

    # Partial updates: only color tile 0 (2x2) and normal tile 1 (2x2).
    (destination / "color-0-low.png").write_bytes(make_png(2, 2, seed=90))
    (destination / "normal-1-low.png").write_bytes(make_png(2, 2, seed=91))
    plan = {
        "tiles": [
            {"material": "tiled-mat", "usage": "color", "position": [0, 0], "file": "color-0-low.png"},
            {"material": "tiled-mat", "usage": "normal", "position": [1, 0], "file": "normal-1-low.png"},
        ]
    }
    (destination / "tile-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return manifest


def main(argv: list[str]) -> int:
    destination = Path(argv[1] if len(argv) > 1 else "samples/tiled")
    write_batch(destination)
    print(destination / "model.gltf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
