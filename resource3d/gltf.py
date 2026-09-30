from __future__ import annotations

import base64
import json
import struct
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from .errors import ImportErrorBundle
from .math3d import Identity, trs
from .models import Asset, Geometry, Material, Mesh, Node, Scene, Skin, Texture

COMPONENT_TYPES = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
TYPE_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
ALLOWED_TOP_LEVEL = {
    "asset", "scene", "scenes", "nodes", "meshes", "materials",
    "accessors", "bufferViews", "buffers", "images", "textures", "skins",
    "samplers", "extensionsUsed", "extensionsRequired",
}
MEDIA_TYPES = {
    "image/png": b"\x89PNG\r\n\x1a\n",
    "image/jpeg": b"\xff\xd8\xff",
}
IMAGE_EXTENSIONS = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
TEXTURE_USAGES = (
    ("baseColorTexture", "base_color"),
    ("metallicRoughnessTexture", "metallic_roughness"),
)
OTHER_TEXTURE_USAGES = (
    ("normalTexture", "normal"),
    ("occlusionTexture", "occlusion"),
    ("emissiveTexture", "emissive"),
)


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

        self.image_docs = self._objects("images")
        self.texture_docs = self._objects("textures")
        self.material_docs = self._objects("materials")
        self.mesh_docs = self._objects("meshes")
        self.skin_docs = self._objects("skins")
        self.images = [self._load_image(index) for index in range(len(self.image_docs))]
        self.textures = [self._texture(index) for index in range(len(self.texture_docs))]
        materials = [self._material(index) for index, item in enumerate(self.material_docs)]
        meshes = [self._mesh(index, materials) for index, item in enumerate(self.mesh_docs)]
        nodes = self._nodes(meshes)
        skins = [self._skin(index) for index in range(len(self.skin_docs))]
        node_skin_ids = self._validate_node_skins(skins)
        if node_skin_ids:
            nodes = self._nodes(meshes, node_skin_ids)
        scenes = self._scenes(nodes)

        if self.errors:
            raise ImportErrorBundle(self.errors)
        objects: dict[str, object] = {}
        for item in [*self.textures, *materials, *self.geometries, *meshes, *skins, *nodes, *scenes]:
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
        for key in ("buffers", "bufferViews", "accessors", "nodes", "meshes", "materials", "scenes", "images", "textures", "skins", "samplers"):
            if key in doc and not isinstance(doc[key], list):
                self.errors.append(f"{key} must be an array")
        for collection_name, unsupported in (
            ("nodes", {"camera", "weights"}),
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
                data = self._read_batch_file(uri, parsed, f"buffers[{index}]")
            else:
                raise ValueError(f"unsupported URI scheme {parsed.scheme!r}")
            if declared != len(data):
                self.errors.append(f"buffers[{index}].byteLength is {declared}, file is {len(data)}")
            return data
        except (KeyError, TypeError, ValueError, OSError, IndexError) as exc:
            self.errors.append(f"buffers[{index}]: {exc}")
            return b""

    def _read_batch_file(self, uri: str, parsed, path: str) -> bytes:
        path_text = unquote(parsed.path if parsed.scheme == "file" else uri)
        if path_text.startswith("/"):
            raise ValueError("absolute URIs are forbidden")
        relative = PurePosixPath(path_text)
        if not relative.parts or ".." in relative.parts:
            raise ValueError("URI escapes batch root")
        candidate = (self.source_path.parent / Path(*relative.parts)).resolve()
        if self.batch_root not in (candidate, *candidate.parents):
            raise ValueError("URI resolves outside batch root")
        try:
            return candidate.read_bytes()
        except OSError as exc:
            raise ValueError(f"missing dependency {candidate.name}: {exc}") from exc

    def _load_image(self, index: int) -> Texture | None:
        try:
            item = self.image_docs[index]
            uri = item.get("uri")
            media_type = item.get("mimeType")
            if uri is None:
                raise ValueError("images without a uri are not supported in this scope")
            parsed = urlparse(uri)
            if parsed.scheme == "data":
                prefix = "data:"
                if ";base64," not in uri:
                    raise ValueError("only base64 data URIs are supported")
                header, encoded = uri.split(",", 1)
                declared_type = header[len(prefix):].split(";", 1)[0]
                data = base64.b64decode(encoded, validate=True)
                media_type = media_type or declared_type
            elif parsed.scheme in {"", "file"} and not parsed.netloc:
                data = self._read_batch_file(uri, parsed, f"images[{index}]")
                if not media_type:
                    suffix = PurePosixPath(unquote(parsed.path if parsed.scheme == "file" else uri)).suffix.lower()
                    media_type = IMAGE_EXTENSIONS.get(suffix)
            else:
                raise ValueError(f"unsupported URI scheme {parsed.scheme!r}")
            if media_type not in MEDIA_TYPES:
                raise ValueError(f"unsupported image media type {media_type!r}")
            if not data.startswith(MEDIA_TYPES[media_type]):
                raise ValueError(f"image bytes do not match {media_type}")
            name = str(item.get("name", f"image-{index}"))
            return Texture(name=name, media_type=media_type, image_bytes=data)
        except (TypeError, ValueError) as exc:
            self.errors.append(f"images[{index}]: {exc}")
            return None

    def _texture(self, index: int) -> Texture | None:
        try:
            item = self.texture_docs[index]
            source = item.get("source")
            if source is None:
                raise ValueError("texture requires an image source")
            image = self.images[int(source)]
            if image is None:
                raise ValueError(f"missing image {source}")
            name = str(item.get("name", image.name))
            return Texture(name=name, media_type=image.media_type, image_bytes=image.image_bytes)
        except (IndexError, TypeError, ValueError) as exc:
            self.errors.append(f"textures[{index}]: {exc}")
            return None

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
        item = self.material_docs[index]
        name = str(item.get("name", f"material-{index}"))
        pbr = item.get("pbrMetallicRoughness", {})
        color = pbr.get("baseColorFactor", [1, 1, 1, 1])
        alpha_mode = item.get("alphaMode", "OPAQUE")
        if not isinstance(pbr, dict) or not isinstance(color, list) or len(color) != 4:
            self.errors.append(f"materials[{index}] has invalid PBR factor")
            color = [1, 1, 1, 1]
        texture_bindings: list[tuple[str, str, int]] = []
        seen_usages: set[str] = set()
        for source_slot, usage in TEXTURE_USAGES:
            slot = pbr.get(source_slot) if isinstance(pbr, dict) else None
            if slot is not None:
                self._bind_texture(slot, usage, index, seen_usages, texture_bindings)
        for source_slot, usage in OTHER_TEXTURE_USAGES:
            slot = item.get(source_slot)
            if slot is not None:
                self._bind_texture(slot, usage, index, seen_usages, texture_bindings)
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
                texture_bindings=tuple(texture_bindings),
            )
        except (TypeError, ValueError):
            self.errors.append(f"materials[{index}] contains a non-numeric factor")
            return Material(name, texture_bindings=tuple(texture_bindings))

    def _bind_texture(
        self,
        slot: object,
        usage: str,
        material_index: int,
        seen_usages: set[str],
        bindings: list[tuple[str, str, int]],
    ) -> None:
        if not isinstance(slot, dict):
            self.errors.append(f"materials[{material_index}] {usage} texture must be an object")
            return
        if usage in seen_usages:
            self.errors.append(f"materials[{material_index}] declares {usage} texture more than once")
            return
        seen_usages.add(usage)
        try:
            texture = self.textures[int(slot["index"])]
        except (KeyError, IndexError, TypeError, ValueError):
            self.errors.append(f"materials[{material_index}] {usage} references missing texture {slot.get('index') if isinstance(slot, dict) else slot!r}")
            return
        if texture is None:
            self.errors.append(f"materials[{material_index}] {usage} references missing texture")
            return
        tex_coord = int(slot.get("texCoord", 0))
        if tex_coord < 0:
            self.errors.append(f"materials[{material_index}] {usage} has a negative texCoord")
        bindings.append((texture.id(), usage, max(tex_coord, 0)))

    def _mesh(self, index: int, materials: list[Material]) -> Mesh:
        item = self.mesh_docs[index]
        name = str(item.get("name", f"mesh-{index}"))
        all_positions: list[float] = []
        all_indices: list[int] = []
        all_normals: list[float] = []
        has_normals = True
        all_joints: list[tuple[int, int, int, int]] = []
        all_weights: list[tuple[float, float, float, float]] = []
        has_skin = True
        bindings: list[tuple[str, int, int]] = []
        face_offset = 0
        skinned_primitives: list[bool] = []
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
            vertex_count = len(positions) // 3
            joints: list[tuple[int, int, int, int]] | None = None
            weights: list[tuple[float, float, float, float]] | None = None
            if "JOINTS_0" in attributes or "WEIGHTS_0" in attributes:
                if "JOINTS_0" not in attributes or "WEIGHTS_0" not in attributes:
                    self.errors.append(f"meshes[{index}] primitive {primitive_index} skinning requires both JOINTS_0 and WEIGHTS_0")
                else:
                    joints = [(int(v[0]), int(v[1]), int(v[2]), int(v[3])) for v in self._read_skin_accessor(int(attributes["JOINTS_0"]), index, primitive_index)]
                    weights = [(float(v[0]), float(v[1]), float(v[2]), float(v[3])) for v in self._read_accessor(int(attributes["WEIGHTS_0"]), "VEC4")]
                    if len(joints) != vertex_count or len(weights) != vertex_count:
                        self.errors.append(f"meshes[{index}] primitive {primitive_index} skinning attributes must match vertex count")
                        joints, weights = None, None
                    else:
                        if any(any(joint < 0 for joint in joint_quad) for joint_quad in joints):
                            self.errors.append(f"meshes[{index}] primitive {primitive_index} has a negative joint index")
                        for vertex_index, weight_quad in enumerate(weights):
                            if any(weight < 0.0 or weight > 1.0 for weight in weight_quad):
                                self.errors.append(f"meshes[{index}] primitive {primitive_index} weight at vertex {vertex_index} is outside [0, 1]")
            vertex_offset = len(all_positions) // 3
            all_positions.extend(positions)
            all_indices.extend(index + vertex_offset for index in indices)
            if normals is None:
                has_normals = False
            elif has_normals:
                all_normals.extend(normals)
            primitive_has_skin = joints is not None and weights is not None
            if not primitive_has_skin:
                has_skin = False
            elif has_skin:
                all_joints.extend(joints)
                all_weights.extend(weights)
            skinned_primitives.append(primitive_has_skin)
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
        if any(skinned_primitives) and not all(skinned_primitives):
            self.errors.append(f"meshes[{index}] mixes skinned and non-skinned primitives")
        geometry = Geometry(
            tuple(all_positions),
            tuple(all_indices),
            tuple(all_normals) if has_normals else None,
            tuple(all_joints) if has_skin else None,
            tuple(all_weights) if has_skin else None,
        )
        self.geometries.append(geometry)
        return Mesh(name, geometry.id(), tuple(bindings))

    def _read_skin_accessor(self, index: int, mesh_index: int, primitive_index: int) -> list[tuple[float, ...]]:
        try:
            accessor = self.accessors[index]
            if not isinstance(accessor, dict) or accessor.get("type") != "VEC4":
                raise ValueError("JOINTS_0 must be VEC4")
            if int(accessor.get("componentType")) not in (5120, 5121, 5122, 5123):
                raise ValueError("JOINTS_0 componentType must be an integer type")
            return self._read_accessor(index, "VEC4")
        except (IndexError, TypeError, ValueError) as exc:
            self.errors.append(f"meshes[{mesh_index}] primitive {primitive_index} JOINTS_0: {exc}")
            return []

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

    def _nodes(self, meshes: list[Mesh], skin_ids: dict[int, str] | None = None) -> list[Node]:
        source_nodes = self.doc.get("nodes", [])
        memo: dict[int, str] = {}
        results: list[Node] = []
        node_ids: dict[int, str] = {}
        skin_ids = skin_ids or {}

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
                skin_id=skin_ids.get(index),
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
        self.node_index_to_result: dict[int, Node] = {}
        for index in node_ids:
            result = next((node for node in results if node.id() == node_ids[index]), None)
            if result is not None:
                self.node_index_to_result[index] = result
        return results

    def _skin(self, index: int) -> Skin | None:
        try:
            item = self.skin_docs[index]
            raw_joints = item.get("joints")
            if not isinstance(raw_joints, list) or not raw_joints:
                raise ValueError("skins require a non-empty joints array")
            joint_node_ids: list[str] = []
            joint_count = len(raw_joints)
            for position, raw_joint in enumerate(raw_joints):
                try:
                    node_id = self.node_ids[int(raw_joint)]
                except (KeyError, TypeError, ValueError):
                    self.errors.append(f"skins[{index}].joints[{position}] references missing node {raw_joint}")
                    return None
                joint_node_ids.append(node_id)
            matrices = self._inverse_bind_matrices(item, index, joint_count)
            if matrices is None:
                return None
            name = str(item.get("name", f"skin-{index}"))
            return Skin(name, tuple(joint_node_ids), matrices)
        except (TypeError, ValueError) as exc:
            self.errors.append(f"skins[{index}]: {exc}")
            return None

    def _inverse_bind_matrices(self, item: dict, skin_index: int, joint_count: int) -> tuple[tuple[float, ...], ...] | None:
        accessor_index = item.get("inverseBindMatrices")
        if accessor_index is None:
            return tuple(Identity for _ in range(joint_count))
        try:
            accessor = self.accessors[int(accessor_index)]
            if not isinstance(accessor, dict) or accessor.get("type") != "MAT4" or int(accessor.get("componentType")) != 5126:
                raise ValueError("inverseBindMatrices must be a float MAT4 accessor")
            if int(accessor.get("count")) != joint_count:
                raise ValueError("inverseBindMatrices count must match joints")
            values = self._read_accessor(int(accessor_index), "MAT4")
            matrices: list[tuple[float, ...]] = []
            for raw in values:
                matrix = tuple(float(value) for value in raw)
                matrices.append(matrix)
            if len(matrices) != joint_count:
                raise ValueError("inverseBindMatrices count must match joints")
            return tuple(matrices)
        except (IndexError, TypeError, ValueError) as exc:
            self.errors.append(f"skins[{skin_index}].inverseBindMatrices: {exc}")
            return None

    def _validate_node_skins(self, skins: list[Skin | None]) -> dict[int, str]:
        source_nodes = self.doc.get("nodes", [])
        skin_ids: dict[int, str] = {}
        for index, item in enumerate(source_nodes if isinstance(source_nodes, list) else []):
            if not isinstance(item, dict) or "skin" not in item:
                continue
            try:
                skin = skins[int(item["skin"])]
            except (IndexError, TypeError, ValueError):
                self.errors.append(f"nodes[{index}] references missing skin {item.get('skin')}")
                continue
            if skin is None:
                self.errors.append(f"nodes[{index}] references missing skin {item.get('skin')}")
                continue
            node = self.node_index_to_result.get(index)
            if node is None:
                continue
            mesh_index = item.get("mesh")
            if mesh_index is None:
                self.errors.append(f"nodes[{index}] references skin {item.get('skin')} without a mesh")
            else:
                try:
                    geometry = self.geometries[int(mesh_index)]
                except (IndexError, TypeError, ValueError):
                    geometry = None
                joint_count = len(skin.joint_node_ids)
                if geometry is not None and geometry.joint_indices is not None and any(
                    joint >= joint_count
                    for quad in geometry.joint_indices
                    for joint in quad
                ):
                    self.errors.append(f"meshes[{mesh_index}] joint index exceeds joints of skins[{item['skin']}]")
            skin_ids[index] = skin.id()
        return skin_ids

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
