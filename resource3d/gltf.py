from __future__ import annotations

import base64
import json
import struct
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from .errors import ImportErrorBundle
from .math3d import Identity, trs
from .models import Asset, Geometry, Material, Mesh, Node, Scene

COMPONENT_TYPES = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
TYPE_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
ALLOWED_TOP_LEVEL = {
    "asset", "scene", "scenes", "nodes", "meshes", "materials",
    "accessors", "bufferViews", "buffers", "extensionsUsed",
    "extensionsRequired",
}


class GltfImporter:
    def __init__(self, source_path: Path, batch_root: Path):
        self.source_path = source_path
        self.batch_root = batch_root.resolve()
        self.errors: list[str] = []

    def load(self) -> tuple[Asset, dict[str, object]]:
        try:
            raw = json.loads(self.source_path.read_text(encoding="utf-8"), parse_constant=self._bad_constant)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ImportErrorBundle([f"{self.source_path}: unreadable or invalid JSON: {exc}"]) from exc

        if not isinstance(raw, dict):
            raise ImportErrorBundle([f"{self.source_path}: glTF root must be an object"])
        self.doc = raw
        self._validate_scope()
        buffer_docs = self.doc.get("buffers", [])
        self.buffers = [self._load_buffer(index) for index in range(len(buffer_docs))] if isinstance(buffer_docs, list) else []
        self.buffer_views = self.doc.get("bufferViews", []) if isinstance(self.doc.get("bufferViews", []), list) else []
        self.accessors = self.doc.get("accessors", []) if isinstance(self.doc.get("accessors", []), list) else []
        self.geometries: list[Geometry] = []

        materials = [self._material(index) for index, item in enumerate(self._objects("materials"))]
        meshes = [self._mesh(index, materials) for index, item in enumerate(self._objects("meshes"))]
        nodes = self._nodes(meshes)
        scenes = self._scenes(nodes)

        if self.errors:
            raise ImportErrorBundle(self.errors)
        objects: dict[str, object] = {}
        for item in [*materials, *self.geometries, *meshes, *nodes, *scenes]:
            objects[item.id()] = item
        asset = Asset("gltf-2.0", tuple(scene.id() for scene in scenes))
        objects[asset.id()] = asset
        return asset, objects

    @staticmethod
    def _bad_constant(value: str):
        raise ValueError(f"invalid JSON constant {value}")

    def _objects(self, name: str) -> list[dict]:
        value = self.doc.get(name, [])
        objects = value if isinstance(value, list) else []
        result = []
        for index, item in enumerate(objects):
            if isinstance(item, dict):
                result.append(item)
            else:
                self.errors.append(f"{name}[{index}] must be an object")
        return result

    def _validate_scope(self) -> None:
        doc = self.doc
        asset = doc.get("asset")
        if not isinstance(asset, dict) or str(asset.get("version", "")) != "2.0":
            self.errors.append("glTF asset.version must be 2.0")
        for key in ("extensionsUsed", "extensionsRequired"):
            if doc.get(key):
                self.errors.append(f"unsupported extension in scope: {doc[key]!r}")
        for key, value in doc.items():
            if key not in ALLOWED_TOP_LEVEL:
                self.errors.append(f"unsupported top-level feature {key!r}")
            elif not isinstance(value, (dict, list, int, str)):
                self.errors.append(f"invalid top-level type for {key}")
        for key in ("buffers", "bufferViews", "accessors", "nodes", "meshes", "materials", "scenes"):
            if key in doc and not isinstance(doc[key], list):
                self.errors.append(f"{key} must be an array")
        for collection_name, unsupported in (
            ("nodes", {"skin", "camera", "weights"}),
            ("meshes", {"weights"}),
        ):
            for index, item in enumerate(doc.get(collection_name, []) if isinstance(doc.get(collection_name, []), list) else []):
                for key in unsupported:
                    if isinstance(item, dict) and key in item:
                        self.errors.append(f"{collection_name}[{index}].{key} is outside supported scope")

    def _load_buffer(self, index: int) -> bytes:
        try:
            buffer = self.doc["buffers"][index]
            if not isinstance(buffer, dict):
                raise TypeError("buffer must be an object")
            uri = buffer.get("uri")
            declared = int(buffer.get("byteLength", 0))
            if uri is None:
                raise ValueError("GLB binary buffers are not supported in this scope")
            parsed = urlparse(uri)
            if parsed.scheme == "data":
                if not uri.startswith("data:application/octet-stream;base64,"):
                    raise ValueError("only data:application/octet-stream;base64 buffers are supported")
                data = base64.b64decode(uri.split(",", 1)[1], validate=True)
            elif parsed.scheme in {"", "file"} and not parsed.netloc:
                path_text = unquote(parsed.path if parsed.scheme == "file" else uri)
                if path_text.startswith("/"):
                    raise ValueError("absolute buffer URIs are forbidden")
                relative = PurePosixPath(path_text)
                if not relative.parts or ".." in relative.parts:
                    raise ValueError("buffer URI escapes batch root")
                candidate = (self.source_path.parent / Path(*relative.parts)).resolve()
                if self.batch_root not in (candidate, *candidate.parents):
                    raise ValueError("buffer URI resolves outside batch root")
                data = candidate.read_bytes()
            else:
                raise ValueError(f"unsupported URI scheme {parsed.scheme!r}")
            if declared != len(data):
                self.errors.append(f"buffers[{index}].byteLength is {declared}, file is {len(data)}")
            return data
        except (KeyError, TypeError, ValueError, OSError, IndexError) as exc:
            self.errors.append(f"buffers[{index}]: {exc}")
            return b""

    def _read_accessor(self, index: int, expected_type: str) -> list[tuple[float, ...]]:
        try:
            accessor = self.accessors[index]
            view = self.buffer_views[int(accessor["bufferView"])]
            if not isinstance(accessor, dict) or not isinstance(view, dict):
                raise TypeError("accessor and bufferView must be objects")
            component_code = accessor["componentType"]
            component_format, component_size = COMPONENT_TYPES[component_code]
            type_name = accessor["type"]
            if type_name != expected_type:
                raise ValueError(f"expected {expected_type}, got {type_name}")
            components = TYPE_COUNTS[type_name]
            count = int(accessor["count"])
            buffer = self.buffers[int(view["buffer"])]
            view_offset = int(view.get("byteOffset", 0))
            view_length = int(view.get("byteLength"))
            accessor_offset = int(accessor.get("byteOffset", 0))
            stride = int(view.get("byteStride", components * component_size))
            if stride < components * component_size:
                raise ValueError("byteStride is smaller than element size")
            if view_offset + view_length > len(buffer):
                raise ValueError("bufferView exceeds buffer")
            start = view_offset + accessor_offset
            if start + stride * count - (stride - components * component_size) > view_offset + view_length:
                raise ValueError("accessor exceeds bufferView")
            values: list[tuple[float, ...]] = []
            for item in range(count):
                element_start = start + item * stride
                raw_values = struct.unpack_from("<" + component_format * components, buffer, element_start)
                values.append(tuple(float(value) for value in raw_values))
            return values
        except (KeyError, TypeError, ValueError, IndexError, struct.error) as exc:
            self.errors.append(f"accessors[{index}]: {exc}")
            return []

    def _material(self, index: int) -> Material:
        item = self._objects("materials")[index]
        name = str(item.get("name", f"material-{index}"))
        pbr = item.get("pbrMetallicRoughness", {})
        color = pbr.get("baseColorFactor", [1, 1, 1, 1])
        alpha_mode = item.get("alphaMode", "OPAQUE")
        if not isinstance(pbr, dict) or not isinstance(color, list) or len(color) != 4:
            self.errors.append(f"materials[{index}] has invalid PBR factor")
            color = [1, 1, 1, 1]
        if isinstance(pbr, dict) and ("baseColorTexture" in pbr or "metallicRoughnessTexture" in pbr):
            self.errors.append(f"materials[{index}] uses textures, which are outside this scope")
        if item.get("normalTexture") or item.get("occlusionTexture") or item.get("emissiveTexture"):
            self.errors.append(f"materials[{index}] uses unsupported texture maps")
        if alpha_mode not in {"OPAQUE", "MASK", "BLEND"}:
            self.errors.append(f"materials[{index}] has invalid alphaMode")
            alpha_mode = "OPAQUE"
        try:
            return Material(
                name=name,
                base_color_factor=tuple(float(value) for value in color),
                metallic_factor=float(pbr.get("metallicFactor", 1)),
                roughness_factor=float(pbr.get("roughnessFactor", 1)),
                alpha_mode=alpha_mode,
                double_sided=bool(item.get("doubleSided", False)),
            )
        except (TypeError, ValueError):
            self.errors.append(f"materials[{index}] contains a non-numeric factor")
            return Material(name)

    def _mesh(self, index: int, materials: list[Material]) -> Mesh:
        item = self._objects("meshes")[index]
        name = str(item.get("name", f"mesh-{index}"))
        all_positions: list[float] = []
        all_indices: list[int] = []
        all_normals: list[float] = []
        has_normals = True
        bindings: list[tuple[str, int, int]] = []
        face_offset = 0
        primitives = item.get("primitives", [])
        if not isinstance(primitives, list):
            self.errors.append(f"meshes[{index}].primitives must be an array")
            primitives = []
        for primitive_index, primitive in enumerate(primitives):
            if not isinstance(primitive, dict):
                self.errors.append(f"meshes[{index}].primitives[{primitive_index}] must be an object")
                continue
            if primitive.get("mode", 4) != 4:
                self.errors.append(f"meshes[{index}] primitive {primitive_index} is not triangles")
            attributes = primitive.get("attributes", {})
            if not isinstance(attributes, dict):
                self.errors.append(f"meshes[{index}] primitive {primitive_index} attributes must be an object")
                continue
            if "POSITION" not in attributes:
                self.errors.append(f"meshes[{index}] primitive {primitive_index} lacks POSITION")
                continue
            positions = [value for vector in self._read_accessor(int(attributes["POSITION"]), "VEC3") for value in vector]
            if "indices" in primitive:
                indices_raw = self._read_accessor(int(primitive["indices"]), "SCALAR")
            else:
                indices_raw = [(float(value),) for value in range(len(positions) // 3)]
            indices = [int(value[0]) for value in indices_raw]
            if any(index < 0 for index in indices):
                self.errors.append(f"meshes[{index}] primitive {primitive_index} has a negative index")
            if indices and max(indices) >= len(positions) // 3:
                self.errors.append(f"meshes[{index}] primitive {primitive_index} index exceeds POSITION bounds")
            normals_raw = self._read_accessor(int(attributes["NORMAL"]), "VEC3") if "NORMAL" in attributes else []
            normals = tuple(value for vector in normals_raw for value in vector) if normals_raw else None
            vertex_offset = len(all_positions) // 3
            all_positions.extend(positions)
            all_indices.extend(index + vertex_offset for index in indices)
            if normals is None:
                has_normals = False
            elif has_normals:
                all_normals.extend(normals)
            material_index = primitive.get("material")
            if material_index is None:
                material = Material(name=f"default-material-{index}-{primitive_index}")
                materials.append(material)
                material_id = material.id()
            else:
                try:
                    material_id = materials[int(material_index)].id()
                except (IndexError, TypeError, ValueError):
                    self.errors.append(f"meshes[{index}] primitive {primitive_index} has missing material {material_index}")
                    material_id = Material("invalid").id()
            face_count = len(indices) // 3
            bindings.append((material_id, face_offset, face_count))
            face_offset += face_count
        if not all_positions:
            self.errors.append(f"meshes[{index}] contains no valid primitive")
        geometry = Geometry(
            tuple(all_positions),
            tuple(all_indices),
            tuple(all_normals) if has_normals else None,
        )
        self.geometries.append(geometry)
        return Mesh(name, geometry.id(), tuple(bindings))

    def _node_transform(self, item: dict, path: str) -> tuple[float, ...]:
        if "matrix" in item and ({"translation", "rotation", "scale"} & item.keys()):
            self.errors.append(f"{path} mixes matrix with TRS")
        if "matrix" in item:
            values = item["matrix"]
            if not isinstance(values, list) or len(values) != 16:
                self.errors.append(f"{path}.matrix must contain 16 numbers")
                return Identity
            try:
                return tuple(float(values[(row % 4) + (row // 4) * 4]) for row in range(16))
            except (TypeError, ValueError):
                self.errors.append(f"{path}.matrix must contain numbers")
                return Identity
        translation = list(item.get("translation", [0.0, 0.0, 0.0]))
        rotation = list(item.get("rotation", [0.0, 0.0, 0.0, 1.0]))
        scale = list(item.get("scale", [1.0, 1.0, 1.0]))
        for key, expected in (("translation", 3), ("rotation", 4), ("scale", 3)):
            value = item.get(key, None)
            if value is not None and (not isinstance(value, list) or len(value) != expected):
                self.errors.append(f"{path}.{key} must contain {expected} numbers")
                if key == "rotation":
                    rotation = [0.0, 0.0, 0.0, 1.0]
                elif key == "translation":
                    translation = [0.0, 0.0, 0.0]
                else:
                    scale = [1.0, 1.0, 1.0]
        try:
            return trs(tuple(translation), tuple(rotation), tuple(scale))
        except (TypeError, ValueError):
            self.errors.append(f"{path} TRS values must be numeric")
            return trs((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0))

    def _nodes(self, meshes: list[Mesh]) -> list[Node]:
        source_nodes = self.doc.get("nodes", [])
        memo: dict[int, str] = {}
        results: list[Node] = []
        node_ids: dict[int, str] = {}

        def build(raw_index: object, stack: tuple[int, ...]) -> str:
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                self.errors.append(f"node reference {raw_index!r} must be an integer")
                return ""
            if index in memo:
                return memo[index]
            if index in stack:
                self.errors.append(f"node cycle at {' -> '.join(str(value) for value in (*stack, index))}")
                return ""
            try:
                item = source_nodes[index]
                if not isinstance(item, dict):
                    raise TypeError
            except (IndexError, TypeError):
                self.errors.append(f"missing node[{index}]")
                return ""
            raw_children = item.get("children", [])
            if not isinstance(raw_children, list):
                self.errors.append(f"nodes[{index}].children must be an array")
                raw_children = []
            children = tuple(build(child, (*stack, index)) for child in raw_children)
            children = tuple(child for child in children if child)
            mesh_id = None
            if "mesh" in item:
                try:
                    mesh_id = meshes[int(item["mesh"])].id()
                except (IndexError, TypeError, ValueError):
                    self.errors.append(f"nodes[{index}] references missing mesh {item.get('mesh')}")
            node = Node(
                name=str(item.get("name", f"node-{index}")),
                transform=self._node_transform(item, f"nodes[{index}]"),
                mesh_id=mesh_id,
                children=children,
            )
            node_id = node.id()
            memo[index] = node_id
            node_ids[index] = node_id
            results.append(node)
            return node_id

        for index, item in enumerate(source_nodes):
            if not isinstance(item, dict):
                self.errors.append(f"nodes[{index}] must be an object")
                continue
            build(index, ())
        self.node_ids = node_ids
        return results

    def _scenes(self, nodes: list[Node]) -> list[Scene]:
        scenes = self._objects("scenes")
        if not scenes:
            return []
        result = []
        for index, item in enumerate(scenes):
            try:
                item = scenes[index]
                roots = []
                raw_nodes = item.get("nodes", [])
                if not isinstance(raw_nodes, list):
                    raise TypeError
                for node_index in raw_nodes:
                    node_id = self.node_ids.get(int(node_index), "")
                    if not node_id:
                        raise IndexError(node_index)
                    roots.append(node_id)
                roots = tuple(roots)
                result.append(Scene(str(item.get("name", f"scene-{index}")), roots))
            except (IndexError, TypeError, ValueError):
                self.errors.append(f"scenes[{index}] references missing node")
        return result
