"""Importer for a deliberately bounded glTF 2.0 subset.

Supported
---------
* ``.gltf`` (with external/embedded buffers and images) and ``.glb``
* scalar/VEC2/VEC3/VEC4 accessors, component types BYTE..FLOAT for vertices
  and UNSIGNED_BYTE/SHORT/INT for indices
* TRIANGLES primitives with POSITION and optional NORMAL / TEXCOORD_0
* PBR metallic-roughness materials (base color factor/texture)
* node hierarchy with matrix or TRS transforms

Everything else (skins, morph targets, sparse accessors, interleaved vertex
buffer views, non-triangle modes, cameras, lights, any extension) is an
explicit error so unsupported data can never silently degrade.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import struct
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from .model import (
    make_geometry,
    make_image,
    make_material,
    make_node,
    make_scene,
)
from .transform import compose_trs, is_normalized_quaternion

GLB_VERSION = 2
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

COMPONENT_FORMATS = {
    5120: ("b", 1, True),
    5121: ("B", 1, True),
    5122: ("h", 2, True),
    5123: ("H", 2, True),
    5125: ("I", 4, False),
    5126: ("f", 4, False),
}
TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
INDEX_COMPONENT_TYPES = {5121, 5123, 5125}
SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg"}


@dataclass
class Issue:
    source: str
    message: str

    def __str__(self) -> str:
        return f"{self.source}: {self.message}"


@dataclass
class LoadedFile:
    source: str
    resources: dict[str, dict] = field(default_factory=dict)
    blobs: dict[str, bytes] = field(default_factory=dict)
    roots: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def load_gltf(path: Path, batch_root: Path) -> LoadedFile:
    """Parse one glTF file, collecting every validation issue found."""
    source = path.name
    loaded = LoadedFile(source=source)
    err = lambda message: loaded.issues.append(Issue(source, message))

    try:
        raw = path.read_bytes()
    except OSError as exc:
        err(f"cannot read file: {exc}")
        return loaded

    try:
        json_bytes, glb_bin = _parse_container(raw)
        gltf = json.loads(json_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        err(str(exc))
        return loaded
    if not isinstance(gltf, dict):
        err("glTF JSON root must be an object")
        return loaded

    _check_top_level_scope(gltf, err)
    if loaded.issues:
        return loaded

    def read_uri(uri: str, what: str) -> bytes | None:
        return _read_external(uri, path, batch_root, glb_bin, loaded.issues, source, what)

    buffers_data = _load_buffers(gltf, glb_bin, read_uri, err)
    images_data = _load_images(gltf, read_uri, buffers_data, err)
    material_ids = _build_materials(gltf, images_data, loaded, err)
    geometry_by_primitive = _build_geometries(gltf, buffers_data, loaded, err)
    loaded.roots = _build_scenes(
        gltf, geometry_by_primitive, material_ids, loaded, err
    )
    return loaded


def _parse_container(raw: bytes) -> tuple[bytes, bytes | None]:
    if raw[:4] != b"glTF":
        return raw, None
    if len(raw) < 12:
        raise ValueError("glb header is truncated")
    _magic, version, total_length = struct.unpack_from("<III", raw, 0)
    if version != GLB_VERSION:
        raise ValueError(f"unsupported glb version {version}")
    if total_length != len(raw):
        raise ValueError("glb length does not match file size")
    offset, json_chunk, bin_chunk = 12, b"", b""
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise ValueError("glb chunk header is truncated")
        length, kind = struct.unpack_from("<II", raw, offset)
        offset += 8
        if offset + length > len(raw):
            raise ValueError("glb chunk is truncated")
        data = raw[offset:offset + length]
        offset += length
        if kind == CHUNK_JSON:
            json_chunk += data
        elif kind == CHUNK_BIN:
            if bin_chunk:
                raise ValueError("multiple BIN chunks in glb")
            bin_chunk = data
        else:
            raise ValueError(f"unknown glb chunk type {kind:#x}")
    if not json_chunk:
        raise ValueError("glb is missing its JSON chunk")
    return json_chunk, (bin_chunk or None)


def _check_top_level_scope(gltf: dict, err) -> None:
    asset = gltf.get("asset")
    version = str(asset.get("version", "")) if isinstance(asset, dict) else ""
    if not version.startswith("2."):
        err(f"unsupported glTF version {version or '<missing>'}; only 2.x")
    for forbidden in ("skins", "cameras", "animations", "lights"):
        if gltf.get(forbidden):
            err(f"{forbidden} are outside the supported subset")
    for used in gltf.get("extensionsUsed", []) or []:
        err(f"extension {used!r} is outside the supported subset")
    for required in gltf.get("extensionsRequired", []) or []:
        err(f"required extension {required!r} is outside the supported subset")
    for index, node in enumerate(gltf.get("nodes", []) or []):
        if node.get("skin") is not None:
            err(f"nodes[{index}].skin is outside the supported subset")
        if "weights" in node or "morphWeights" in node:
            err(f"nodes[{index}] morph weights are outside the supported subset")
    for mesh_index, mesh in enumerate(gltf.get("meshes", []) or []):
        for prim_index, primitive in enumerate(mesh.get("primitives", [])):
            if "targets" in primitive:
                err(f"meshes[{mesh_index}].primitives[{prim_index}] morph targets unsupported")
            if primitive.get("mode", 4) != 4:
                err(f"meshes[{mesh_index}].primitives[{prim_index}] must be TRIANGLES")
    for index, accessor in enumerate(gltf.get("accessors", []) or []):
        if "sparse" in accessor:
            err(f"accessors[{index}] sparse storage is not supported")


def _read_external(
    uri: str,
    gltf_path: Path,
    batch_root: Path,
    glb_bin: bytes | None,
    issues: list[Issue],
    source: str,
    what: str,
) -> bytes | None:
    parsed = urllib.parse.urlsplit(uri)
    if parsed.scheme in {"http", "https", "ftp", "file"}:
        issues.append(Issue(source, f"{what} URI must be inside the batch: {uri!r}"))
        return None
    if parsed.scheme == "data":
        try:
            header, encoded = uri.split(",", 1)
            if ";base64" not in header:
                raise ValueError("only base64 data URIs are supported")
            return base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            issues.append(Issue(source, f"invalid data URI for {what}: {exc}"))
            return None
    if parsed.scheme:
        issues.append(Issue(source, f"unsupported URI scheme for {what}: {parsed.scheme!r}"))
        return None
    if os.path.isabs(uri):
        issues.append(Issue(source, f"{what} URI must stay inside the batch: {uri!r}"))
        return None
    candidate = (gltf_path.parent / urllib.parse.unquote(uri)).resolve()
    root = batch_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        issues.append(Issue(source, f"{what} URI escapes the batch directory: {uri!r}"))
        return None
    if not candidate.is_file():
        issues.append(Issue(source, f"missing dependency: {candidate.relative_to(root)}"))
        return None
    try:
        return candidate.read_bytes()
    except OSError as exc:
        issues.append(Issue(source, f"cannot read {what} {uri!r}: {exc}"))
        return None


def _load_buffers(gltf: dict, glb_bin: bytes | None, read_uri, err) -> list[bytes | None]:
    buffers: list[bytes | None] = []
    for index, buffer in enumerate(gltf.get("buffers", []) or []):
        uri = buffer.get("uri")
        if uri is None:
            data = glb_bin if index == 0 else None
            if data is None:
                err(f"buffers[{index}] has no uri and no GLB binary chunk")
        else:
            data = read_uri(uri, f"buffers[{index}]")
        if data is not None and "byteLength" in buffer:
            if len(data) < int(buffer["byteLength"]):
                err(f"buffers[{index}] is shorter than declared byteLength")
                data = None
        buffers.append(data)
    return buffers


def _buffer_view(gltf: dict, buffers: list[bytes | None], index: int, what: str, err):
    views = gltf.get("bufferViews", []) or []
    if not isinstance(index, int) or not 0 <= index < len(views):
        err(f"{what} references missing bufferView {index}")
        return None
    view = views[index]
    buffer_index = view.get("buffer", -1)
    if not 0 <= buffer_index < len(buffers) or buffers[buffer_index] is None:
        err(f"{what} bufferView[{index}] buffer is unavailable")
        return None
    if view.get("byteStride"):
        err(f"{what} bufferView[{index}] interleaved stride is not supported")
        return None
    data = buffers[buffer_index]
    start = int(view.get("byteOffset", 0))
    end = start + int(view.get("byteLength", 0))
    if end > len(data):
        err(f"{what} bufferView[{index}] exceeds its buffer")
        return None
    return data[start:end]


def _read_accessor(gltf, buffers, index: int, what: str, err):
    accessors = gltf.get("accessors", []) or []
    if not isinstance(index, int) or not 0 <= index < len(accessors):
        err(f"{what} references missing accessor {index}")
        return None
    accessor = accessors[index]
    component_type = accessor.get("componentType")
    if component_type not in COMPONENT_FORMATS:
        err(f"{what} accessor[{index}] has unsupported componentType {component_type}")
        return None
    fmt, size, _normalizable = COMPONENT_FORMATS[component_type]
    accessor_type = accessor.get("type")
    components = TYPE_COMPONENTS.get(accessor_type)
    if components is None:
        err(f"{what} accessor[{index}] has unsupported type {accessor_type!r}")
        return None
    count = int(accessor.get("count", 0))
    view_bytes = _buffer_view(
        gltf, buffers, int(accessor.get("bufferView", -1)), what, err
    )
    if view_bytes is None:
        return None
    offset = int(accessor.get("byteOffset", 0))
    item_size = size * components
    needed = offset + item_size * count
    if needed > len(view_bytes):
        err(f"{what} accessor[{index}] reads past its bufferView")
        return None
    if count == 0:
        err(f"{what} accessor[{index}] is empty")
        return None
    values: list[list[float] | int] = []
    for element in range(count):
        base = offset + element * item_size
        raw = struct.unpack_from("<" + fmt * components, view_bytes, base)
        if components == 1:
            values.append(raw[0])
        else:
            values.append([float(item) for item in raw])
    min_declared = accessor.get("min")
    max_declared = accessor.get("max")
    if min_declared is not None and len(min_declared) != components:
        err(f"{what} accessor[{index}] min has wrong component count")
    if max_declared is not None and len(max_declared) != components:
        err(f"{what} accessor[{index}] max has wrong component count")
    return values


def _build_geometries(gltf, buffers, loaded: LoadedFile, err) -> dict[tuple[int, int], str]:
    geometry_ids: dict[tuple[int, int], str] = {}
    for mesh_index, mesh in enumerate(gltf.get("meshes", []) or []):
        for prim_index, primitive in enumerate(mesh.get("primitives", [])):
            what = f"meshes[{mesh_index}].primitives[{prim_index}]"
            attributes = primitive.get("attributes", {})
            if "POSITION" not in attributes:
                err(f"{what} is missing POSITION")
                continue
            positions = _read_accessor(
                gltf, buffers, attributes["POSITION"], f"{what}.POSITION", err
            )
            normals = None
            uvs = None
            if positions is None:
                continue
            if any(len(point) != 3 for point in positions):
                err(f"{what}.POSITION must be VEC3")
                continue
            if "NORMAL" in attributes:
                normals = _read_accessor(
                    gltf, buffers, attributes["NORMAL"], f"{what}.NORMAL", err
                )
                if normals is not None and any(len(n) != 3 for n in normals):
                    err(f"{what}.NORMAL must be VEC3")
                    normals = None
            if "TEXCOORD_0" in attributes:
                uvs = _read_accessor(
                    gltf, buffers, attributes["TEXCOORD_0"], f"{what}.TEXCOORD_0", err
                )
                if uvs is not None and any(len(uv) != 2 for uv in uvs):
                    err(f"{what}.TEXCOORD_0 must be VEC2")
                    uvs = None
            index_accessor = primitive.get("indices")
            if index_accessor is None:
                indices = list(range(len(positions)))
            else:
                raw_indices = _read_accessor(
                    gltf, buffers, index_accessor, f"{what}.indices", err
                ) or []
                accessor = gltf["accessors"][index_accessor]
                if accessor.get("componentType") not in INDEX_COMPONENT_TYPES:
                    err(f"{what}.indices must use an unsigned integer component type")
                    continue
                indices = [int(value) for value in raw_indices]
            if len(indices) % 3:
                err(f"{what} index count must be divisible by 3 for TRIANGLES")
                continue
            triangles = [indices[i:i + 3] for i in range(0, len(indices), 3)]
            try:
                geometry_id, payload = make_geometry(
                    positions=positions,
                    triangles=triangles,
                    normals=normals,
                    uvs=uvs,
                )
            except ValueError as exc:
                err(f"{what}: {exc}")
                continue
            geometry_ids[(mesh_index, prim_index)] = geometry_id
            loaded.resources.setdefault(geometry_id, payload)
    return geometry_ids


def _load_images(gltf, read_uri, buffers, err) -> list:
    image_sources: list = []
    images = gltf.get("images", []) or []
    for index, image in enumerate(images):
        media_type = image.get("mimeType")
        if "bufferView" in image:
            data = _buffer_view(gltf, buffers, image["bufferView"], f"images[{index}]", err)
            if media_type is None:
                err(f"images[{index}] bufferView image requires mimeType")
                image_sources.append(None)
                continue
        else:
            uri = image.get("uri")
            if not uri:
                err(f"images[{index}] has neither uri nor bufferView")
                image_sources.append(None)
                continue
            data = read_uri(uri, f"images[{index}]")
            if media_type is None:
                media_type = "image/png" if uri.lower().endswith(".png") else "image/jpeg"
        if data is None:
            image_sources.append(None)
            continue
        if media_type not in SUPPORTED_IMAGE_TYPES:
            err(f"images[{index}] unsupported media type {media_type!r}")
            image_sources.append(None)
            continue
        image_sources.append((media_type, data))
    return image_sources


def _build_materials(gltf, images_data, loaded: LoadedFile, err) -> list:
    material_ids: list = []
    textures = gltf.get("textures", []) or []
    for index, material in enumerate(gltf.get("materials", []) or []):
        pbr = material.get("pbrMetallicRoughness", {}) or {}
        color = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
        base_color_image = None
        texture_info = pbr.get("baseColorTexture")
        if texture_info is not None:
            if texture_info.get("texCoord", 0) != 0:
                err(f"materials[{index}] non-zero baseColorTexture texCoord unsupported")
            texture_index = texture_info.get("index", -1)
            if not 0 <= texture_index < len(textures):
                err(f"materials[{index}] baseColorTexture references missing texture")
            else:
                image_index = textures[texture_index].get("source", -1)
            if texture_info is not None and 0 <= texture_index < len(textures):
                source = images_data[image_index] if 0 <= image_index < len(images_data) else None
                if source:
                    media_type, data = source
                    image_id, image_payload = make_image(media_type=media_type, data=data)
                    loaded.resources.setdefault(image_id, image_payload)
                    loaded.blobs.setdefault(image_payload["blobId"], data)
                    base_color_image = image_id
                elif texture_info is not None:
                    err(f"materials[{index}] baseColorTexture image is unavailable")
        try:
            material_id, payload = make_material(
                base_color_factor=color,
                base_color_image=base_color_image,
                metallic_factor=pbr.get("metallicFactor", 1.0),
                roughness_factor=pbr.get("roughnessFactor", 1.0),
                name=material.get("name"),
            )
        except (ValueError, TypeError) as exc:
            err(f"materials[{index}]: {exc}")
            material_ids.append(None)
            continue
        material_ids.append(material_id)
        loaded.resources.setdefault(material_id, payload)
    return material_ids


def _node_matrix(node: dict, label: str, err):
    has_trs = any(key in node for key in ("translation", "rotation", "scale"))
    if "matrix" in node and has_trs:
        err(f"{label} must not define both matrix and TRS")
        return None
    if "matrix" in node:
        values = node["matrix"]
        if not isinstance(values, list) or len(values) != 16:
            err(f"{label}.matrix must contain 16 numbers")
            return None
        try:
            return [float(value) for value in values]
        except (TypeError, ValueError):
            err(f"{label}.matrix contains non-numeric entries")
            return None
    translation = node.get("translation", [0.0, 0.0, 0.0])
    rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    scale = node.get("scale", [1.0, 1.0, 1.0])
    try:
        translation = [float(v) for v in translation]
        rotation = [float(v) for v in rotation]
        scale = [float(v) for v in scale]
    except (TypeError, ValueError):
        err(f"{label} TRS contains non-numeric entries")
        return None
    if len(translation) != 3 or len(scale) != 3 or len(rotation) != 4:
        err(f"{label} TRS components have the wrong length")
        return None
    if not is_normalized_quaternion(rotation):
        err(f"{label}.rotation must be a unit quaternion")
        return None
    return compose_trs(translation, rotation, scale)


def _build_scenes(gltf, geometry_by_primitive, material_ids, loaded: LoadedFile, err):
    nodes = gltf.get("nodes", []) or []
    meshes = gltf.get("meshes", []) or []
    materials = gltf.get("materials", []) or []
    node_ids: dict[int, str] = {}

    def mesh_bindings(node_index: int):
        mesh_index = nodes[node_index].get("mesh")
        bindings = []
        if mesh_index is None:
            return bindings
        if not 0 <= mesh_index < len(meshes):
            err(f"nodes[{node_index}] references missing meshes[{mesh_index}]")
            return bindings
        for prim_index, primitive in enumerate(meshes[mesh_index].get("primitives", [])):
            geometry_id = geometry_by_primitive.get((mesh_index, prim_index))
            if geometry_id is None:
                err(f"nodes[{node_index}] geometry for primitive {prim_index} is unavailable")
                continue
            material_index = primitive.get("material")
            if material_index is None:
                material_id, payload = make_material(name="r3d-default")
                loaded.resources.setdefault(material_id, payload)
            elif not 0 <= material_index < len(materials):
                err(f"nodes[{node_index}] references missing materials[{material_index}]")
                continue
            else:
                material_id = material_ids[material_index]
                if material_id is None:
                    err(f"nodes[{node_index}] material {material_index} is unavailable")
                    continue
            bindings.append({"geometry": geometry_id, "material": material_id})
        return bindings

    def build_node(index: int, trail: set[int]):
        if index in node_ids:
            return node_ids[index]
        if index in trail:
            err(f"node hierarchy contains a cycle at nodes[{index}]")
            return None
        node = nodes[index]
        child_ids = []
        for child in node.get("children", []) or []:
            if not isinstance(child, int) or not 0 <= child < len(nodes):
                err(f"nodes[{index}] has invalid child reference {child!r}")
                continue
            child_id = build_node(child, trail | {index})
            if child_id is not None:
                child_ids.append(child_id)
        matrix = _node_matrix(node, f"nodes[{index}]", err)
        if matrix is None:
            return None
        try:
            node_id, payload = make_node(
                matrix=matrix,
                children=child_ids,
                meshes=mesh_bindings(index),
                name=node.get("name"),
            )
        except ValueError as exc:
            err(f"nodes[{index}]: {exc}")
            return None
        node_ids[index] = node_id
        loaded.resources.setdefault(node_id, payload)
        return node_id

    scene_ids = []
    for scene_index, scene in enumerate(gltf.get("scenes", []) or []):
        roots = []
        for node_index in scene.get("nodes", []) or []:
            if not isinstance(node_index, int) or not 0 <= node_index < len(nodes):
                err(f"scenes[{scene_index}] has invalid node reference {node_index!r}")
                continue
            node_id = build_node(node_index, set())
            if node_id is not None:
                roots.append(node_id)
        try:
            scene_id, payload = make_scene(roots=roots, name=scene.get("name"))
        except ValueError as exc:
            err(f"scenes[{scene_index}]: {exc}")
            continue
        scene_ids.append(scene_id)
        loaded.resources.setdefault(scene_id, payload)
    return scene_ids
