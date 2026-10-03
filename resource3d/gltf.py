from __future__ import annotations

import base64
import json
import struct
from dataclasses import replace
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from .errors import ImportErrorBundle
from .math3d import Identity, trs
from .models import Asset, Geometry, Image, Material, Mesh, Node, Scene, Skin, Texture

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
    "extensionsUsed", "extensionsRequired",
}
IMAGE_MIME_TYPES = {"image/png", "image/jpeg"}


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

        self.images = [self._image(index) for index, item in enumerate(self._objects("images"))]
        self.textures = [self._texture(index) for index, item in enumerate(self._objects("textures"))]
        self.bound_textures: dict[str, Texture] = {}
        materials = [self._material(index) for index, item in enumerate(self._objects("materials"))]
        meshes = [self._mesh(index, materials) for index, item in enumerate(self._objects("meshes"))]
        nodes = self._nodes(meshes)
        skins = self._skins()
        nodes = self._attach_skins(nodes, meshes, skins)
        scenes = self._scenes(nodes)

        if self.errors:
            raise ImportErrorBundle(self.errors)
        objects: dict[str, object] = {}
        all_textures = list(self.bound_textures.values())
        for texture in self.textures:
            if texture.image_id and texture.id() not in self.bound_textures:
                all_textures.append(texture)
        for item in [*self.images, *all_textures, *materials, *self.geometries, *meshes, *skins, *nodes, *scenes]:
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
        for key in (
            "buffers", "bufferViews", "accessors", "nodes", "meshes", "materials",
            "scenes", "images", "textures", "skins",
        ):
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

    def _resolve_batch_file(self, uri: str, label: str) -> bytes:
        parsed = urlparse(uri)
        if parsed.scheme not in {"", "file"} or parsed.netloc:
            raise ValueError(f"unsupported URI scheme {parsed.scheme!r}")
        path_text = unquote(parsed.path if parsed.scheme == "file" else uri)
        if path_text.startswith("/"):
            raise ValueError("absolute URIs are forbidden")
        relative = PurePosixPath(path_text)
        if not relative.parts or ".." in relative.parts:
            raise ValueError(f"{label} URI escapes batch root")
        candidate = (self.source_path.parent / Path(*relative.parts)).resolve()
        if self.batch_root not in (candidate, *candidate.parents):
            raise ValueError(f"{label} URI resolves outside batch root")
        return candidate.read_bytes()

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
            else:
                data = self._resolve_batch_file(uri, "buffer")
            if declared != len(data):
                self.errors.append(f"buffers[{index}].byteLength is {declared}, file is {len(data)}")
            return data
        except (KeyError, TypeError, ValueError, OSError, IndexError) as exc:
            self.errors.append(f"buffers[{index}]: {exc}")
            return b""

    def _image(self, index: int) -> Image:
        fallback = Image(f"image-{index}", "image/png", b"")
        try:
            item = self._objects("images")[index]
            name = str(item.get("name", f"image-{index}"))
            mime_type = str(item.get("mimeType", ""))
            if "uri" in item:
                uri = item["uri"]
                parsed = urlparse(uri)
                if parsed.scheme == "data":
                    prefix, _, encoded = uri.partition(",")
                    if not prefix.startswith("data:") or ";base64" not in prefix:
                        raise ValueError("only base64 data: image URIs are supported")
                    declared_mime = prefix[len("data:"):].split(";", 1)[0]
                    mime_type = mime_type or declared_mime
                    payload = base64.b64decode(encoded, validate=True)
                else:
                    if not mime_type:
                        suffix = PurePosixPath(unquote(parsed.path if parsed.scheme == "file" else uri)).suffix.lower()
                        mime_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(suffix, "")
                    payload = self._resolve_batch_file(uri, "image")
            elif "bufferView" in item:
                view_index = int(item["bufferView"])
                view = self.buffer_views[view_index]
                if not isinstance(view, dict):
                    raise TypeError("image bufferView must be an object")
                buffer = self.buffers[int(view["buffer"])]
                view_offset = int(view.get("byteOffset", 0))
                view_length = int(view["byteLength"])
                if view_offset + view_length > len(buffer):
                    raise ValueError("image bufferView exceeds buffer")
                payload = buffer[view_offset:view_offset + view_length]
            else:
                raise ValueError("image needs uri or bufferView")
            if mime_type not in IMAGE_MIME_TYPES:
                raise ValueError(f"unsupported image MIME type {mime_type!r}")
            if not payload:
                raise ValueError("image payload is empty")
            return Image(name, mime_type, payload)
        except (KeyError, TypeError, ValueError, OSError, IndexError, struct.error) as exc:
            self.errors.append(f"images[{index}]: {exc}")
            return fallback

    def _texture(self, index: int) -> Texture:
        fallback = Texture(f"texture-{index}", "", "color")
        try:
            item = self._objects("textures")[index]
            name = str(item.get("name", f"texture-{index}"))
            image_index = int(item["source"])
            image = self.images[image_index]
            if not image.payload:
                raise ValueError(f"references broken images[{image_index}]")
            tile = self._tile_spec(item.get("extras"), name)
            return Texture(
                name,
                image.id(),
                "color",
                tile_position=tile[0],
                tile_layout=tile[1],
                tile_edges=tile[2],
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            self.errors.append(f"textures[{index}]: {exc}")
            return fallback

    def _tile_spec(
        self, extras: object, label: str
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None, tuple[tuple[str, ...], ...] | None]:
        if not isinstance(extras, dict) or not isinstance(extras.get("tile"), dict):
            return None, None, None
        tile = extras["tile"]
        position_raw = tile.get("position")
        layout_raw = tile.get("layout")
        if not isinstance(position_raw, list) or len(position_raw) != 2:
            raise ValueError(f"{label} tile position must be [column, row]")
        if not isinstance(layout_raw, list) or len(layout_raw) != 2:
            raise ValueError(f"{label} tile layout must be [columns, rows]")
        column, row = int(position_raw[0]), int(position_raw[1])
        columns, rows = int(layout_raw[0]), int(layout_raw[1])
        if columns <= 0 or rows <= 0 or not (0 <= column < columns) or not (0 <= row < rows):
            raise ValueError(f"{label} tile position {column},{row} outside layout {columns}x{rows}")
        edges_raw = tile.get("edges")
        edges: tuple[tuple[str, ...], ...] | None = None
        if edges_raw is not None:
            if not isinstance(edges_raw, list) or len(edges_raw) != 4 or not all(
                isinstance(side, list) and all(isinstance(tag, str) for tag in side) for side in edges_raw
            ):
                raise ValueError(f"{label} tile edges must be four lists of edge tags (left, right, top, bottom)")
            edges = tuple(tuple(side) for side in edges_raw)
        return (column, row), (columns, rows), edges

    def _texture_binding(self, container: dict, key: str, material_index: int, usage: str) -> str | None:
        raw = container.get(key)
        if raw is None:
            return None
        try:
            if not isinstance(raw, dict):
                raise TypeError("texture info must be an object")
            texture_index = int(raw["index"])
            base = self.textures[texture_index]
            if not base.image_id:
                raise ValueError(f"references broken textures[{texture_index}]")
            tex_coord = int(raw.get("texCoord", 0))
            if tex_coord < 0:
                raise ValueError("texCoord must be non-negative")
            bound = Texture(
                base.name,
                base.image_id,
                usage,  # type: ignore[arg-type]
                tex_coord,
                base.tile_position,
                base.tile_layout,
                base.tile_edges,
            )
            self.bound_textures[bound.id()] = bound
            return bound.id()
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            self.errors.append(f"materials[{material_index}].{key}: {exc}")
            return None

    def _tile_texture_slots(self, material_index: int, item: dict) -> dict[str, tuple[str, ...]]:
        extras = item.get("extras")
        if not isinstance(extras, dict) or not isinstance(extras.get("tiles"), dict):
            return {}
        slots: dict[str, tuple[str, ...]] = {}
        valid_usages = {"color", "data", "normal", "occlusion", "emissive"}
        for usage, entries in extras["tiles"].items():
            if usage not in valid_usages:
                self.errors.append(f"materials[{material_index}].extras.tiles: unknown usage {usage!r}")
                continue
            if not isinstance(entries, list) or not entries:
                self.errors.append(f"materials[{material_index}].extras.tiles.{usage} must be a non-empty list")
                continue
            bound_ids: list[str] = []
            specs: list[tuple[tuple[int, int], tuple[int, int], tuple[tuple[str, ...], ...] | None, str]] = []
            tex_coord: int | None = None
            for slot, entry in enumerate(entries):
                label = f"materials[{material_index}].extras.tiles.{usage}[{slot}]"
                try:
                    if not isinstance(entry, dict):
                        raise TypeError("tile entry must be an object")
                    texture_index = int(entry["index"])
                    base = self.textures[texture_index]
                    if not base.image_id:
                        raise ValueError(f"references broken textures[{texture_index}]")
                    slot_coord = int(entry.get("texCoord", base.tex_coord))
                    if slot_coord < 0:
                        raise ValueError("texCoord must be non-negative")
                    if tex_coord is None:
                        tex_coord = slot_coord
                    elif slot_coord != tex_coord:
                        raise ValueError("all tiles of one usage must share texCoord")
                    tile = self._tile_spec(entry.get("extras"), label)
                    position = tile[0] or base.tile_position
                    layout = tile[1] or base.tile_layout
                    edges = tile[2] if tile[2] is not None else base.tile_edges
                    if position is None or layout is None:
                        raise ValueError("tile needs position and layout")
                    bound = Texture(base.name, base.image_id, usage, slot_coord, position, layout, edges)  # type: ignore[arg-type]
                    self.bound_textures[bound.id()] = bound
                    bound_ids.append(bound.id())
                    specs.append((position, layout, edges, label))
                except (KeyError, TypeError, ValueError, IndexError) as exc:
                    self.errors.append(str(exc))
            if not specs:
                continue
            layout = specs[0][1]
            positions = {spec[0] for spec in specs}
            if any(spec[1] != layout for spec in specs):
                self.errors.append(f"materials[{material_index}].extras.tiles.{usage}: tiles disagree on layout")
            if len(positions) != len(specs):
                self.errors.append(f"materials[{material_index}].extras.tiles.{usage}: duplicate tile positions")
            columns, rows = layout
            expected = {(column, row) for column in range(columns) for row in range(rows)}
            if positions != expected:
                missing = sorted(expected - positions)
                self.errors.append(
                    f"materials[{material_index}].extras.tiles.{usage}: incomplete tile set, missing {missing}"
                )
            by_position = {spec[0]: spec for spec in specs}
            for (column, row), _layout, edges, label in specs:
                if edges is None:
                    continue
                for side_a, side_b, dcolumn, drow in (
                    (1, 0, 1, 0),
                    (3, 2, 0, 1),
                ):
                    neighbor_pos = (column + dcolumn, row + drow)
                    if neighbor_pos not in positions:
                        continue
                    neighbor_edges = by_position[neighbor_pos][2]
                    if neighbor_edges is None:
                        continue
                    if tuple(edges[side_a]) != tuple(neighbor_edges[side_b]):
                        self.errors.append(
                            f"{label}: edge tags do not match tile {neighbor_pos[0]},{neighbor_pos[1]} across the shared seam"
                        )
            slots[usage] = tuple(bound_ids)
        return slots

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
                if expected_type == "MAT4":
                    # glTF matrices are column-major; the internal model is row-major.
                    raw_values = tuple(raw_values[(row % 4) + (row // 4) * 4] for row in range(16))
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
        if alpha_mode not in {"OPAQUE", "MASK", "BLEND"}:
            self.errors.append(f"materials[{index}] has invalid alphaMode")
            alpha_mode = "OPAQUE"
        texture_ids: dict[str, str | None] = {
            "color": None,
            "data": None,
            "normal": None,
            "occlusion": None,
            "emissive": None,
        }
        if isinstance(pbr, dict):
            texture_ids["color"] = self._texture_binding(pbr, "baseColorTexture", index, "color")
            texture_ids["data"] = self._texture_binding(pbr, "metallicRoughnessTexture", index, "data")
            texture_ids["normal"] = self._texture_binding(item, "normalTexture", index, "normal")
            texture_ids["occlusion"] = self._texture_binding(item, "occlusionTexture", index, "occlusion")
            texture_ids["emissive"] = self._texture_binding(item, "emissiveTexture", index, "emissive")
        tile_slots = self._tile_texture_slots(index, item)
        for usage in tile_slots:
            texture_ids[usage] = None
        emissive_factor = item.get("emissiveFactor", [0.0, 0.0, 0.0])
        if not isinstance(emissive_factor, list) or len(emissive_factor) != 3:
            self.errors.append(f"materials[{index}] has invalid emissiveFactor")
            emissive_factor = [0.0, 0.0, 0.0]
        try:
            return Material(
                name=name,
                base_color_factor=tuple(float(value) for value in color),
                metallic_factor=float(pbr.get("metallicFactor", 1)),
                roughness_factor=float(pbr.get("roughnessFactor", 1)),
                alpha_mode=alpha_mode,
                double_sided=bool(item.get("doubleSided", False)),
                color_texture_id=texture_ids["color"],
                data_texture_id=texture_ids["data"],
                normal_texture_id=texture_ids["normal"],
                occlusion_texture_id=texture_ids["occlusion"],
                emissive_texture_id=texture_ids["emissive"],
                emissive_factor=tuple(float(value) for value in emissive_factor),
                normal_scale=self._texture_scale(item, "normalTexture", "scale", index, 1.0),
                occlusion_strength=self._texture_scale(item, "occlusionTexture", "strength", index, 1.0),
                tile_textures=tuple(sorted(tile_slots.items())),
            )
        except (TypeError, ValueError):
            self.errors.append(f"materials[{index}] contains a non-numeric factor")
            return Material(name)

    def _texture_scale(self, item: dict, key: str, field: str, material_index: int, default: float) -> float:
        info = item.get(key)
        if not isinstance(info, dict) or field not in info:
            return default
        try:
            return float(info[field])
        except (TypeError, ValueError):
            self.errors.append(f"materials[{material_index}].{key}.{field} must be numeric")
            return default

    def _mesh(self, index: int, materials: list[Material]) -> Mesh:
        item = self._objects("meshes")[index]
        name = str(item.get("name", f"mesh-{index}"))
        all_positions: list[float] = []
        all_indices: list[int] = []
        all_normals: list[float] = []
        all_joints: list[int] = []
        all_weights: list[float] = []
        has_normals = True
        has_skin = True
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
            vertex_count = len(positions) // 3
            if "indices" in primitive:
                indices_raw = self._read_accessor(int(primitive["indices"]), "SCALAR")
            else:
                indices_raw = [(float(value),) for value in range(vertex_count)]
            indices = [int(value[0]) for value in indices_raw]
            if any(index < 0 for index in indices):
                self.errors.append(f"meshes[{index}] primitive {primitive_index} has a negative index")
            if indices and max(indices) >= vertex_count:
                self.errors.append(f"meshes[{index}] primitive {primitive_index} index exceeds POSITION bounds")
            normals_raw = self._read_accessor(int(attributes["NORMAL"]), "VEC3") if "NORMAL" in attributes else []
            normals = tuple(value for vector in normals_raw for value in vector) if normals_raw else None
            joint_attribute = attributes.get("JOINTS_0")
            weight_attribute = attributes.get("WEIGHTS_0")
            joints: tuple[int, ...] | None = None
            weights: tuple[float, ...] | None = None
            if joint_attribute is not None or weight_attribute is not None:
                if joint_attribute is None or weight_attribute is None:
                    self.errors.append(
                        f"meshes[{index}] primitive {primitive_index} needs JOINTS_0 and WEIGHTS_0 together"
                    )
                else:
                    try:
                        joint_component = self.accessors[int(joint_attribute)]["componentType"]
                    except (IndexError, TypeError, KeyError):
                        joint_component = None
                    if joint_component not in (5120, 5121, 5122, 5123):
                        self.errors.append(
                            f"meshes[{index}] primitive {primitive_index} JOINTS_0 must use an integer component type"
                        )
                    joints_raw = self._read_accessor(int(joint_attribute), "VEC4")
                    weights_raw = self._read_accessor(int(weight_attribute), "VEC4")
                    if joints_raw and weights_raw and (len(joints_raw) != vertex_count or len(weights_raw) != vertex_count):
                        self.errors.append(
                            f"meshes[{index}] primitive {primitive_index} joint/weight count does not match POSITION"
                        )
                    joints = tuple(int(value) for vector in joints_raw for value in vector)
                    weights = tuple(float(value) for vector in weights_raw for value in vector)
                    if joints and any(joint < 0 for joint in joints):
                        self.errors.append(f"meshes[{index}] primitive {primitive_index} has a negative joint index")
                    for vertex in range(vertex_count):
                        vertex_weights = weights[vertex * 4:vertex * 4 + 4]
                        if vertex_weights and any(weight < 0.0 or weight > 1.0 for weight in vertex_weights):
                            self.errors.append(
                                f"meshes[{index}] primitive {primitive_index} has out-of-range skin weights"
                            )
                            break
                    total = sum(weights) if weights else 0.0
                    if weights and not abs(total - vertex_count) <= 1e-5 * max(1, vertex_count):
                        self.errors.append(
                            f"meshes[{index}] primitive {primitive_index} skin weights must sum to 1 per vertex"
                        )
            vertex_offset = len(all_positions) // 3
            all_positions.extend(positions)
            all_indices.extend(index + vertex_offset for index in indices)
            if normals is None:
                has_normals = False
            elif has_normals:
                all_normals.extend(normals)
            if joints is None or weights is None:
                has_skin = False
            elif has_skin:
                all_joints.extend(joints)
                all_weights.extend(weights)
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
            tuple(all_joints) if has_skin else None,
            tuple(all_weights) if has_skin else None,
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

    def _skins(self) -> list[Skin]:
        skins = self._objects("skins")
        result: list[Skin] = []
        self.skin_specs: dict[int, tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]] = {}
        for index, item in enumerate(skins):
            name = str(item.get("name", f"skin-{index}"))
            raw_joints = item.get("joints")
            if not isinstance(raw_joints, list) or not raw_joints:
                self.errors.append(f"skins[{index}].joints must be a non-empty array")
                result.append(Skin(name, (), ()))
                continue
            joint_ids: list[str] = []
            joint_refs_failed = False
            for joint_ref in raw_joints:
                try:
                    joint_index = int(joint_ref)
                    joint_id = self.node_ids[joint_index]
                except (TypeError, ValueError, KeyError):
                    self.errors.append(f"skins[{index}] references missing joint node {joint_ref}")
                    joint_refs_failed = True
                    break
                joint_ids.append(joint_id)
            if joint_refs_failed:
                result.append(Skin(name, (), ()))
                continue
            matrices: list[tuple[float, ...]] = []
            if "inverseBindMatrices" in item:
                raw_matrices = self._read_accessor(int(item["inverseBindMatrices"]), "MAT4")
                if len(raw_matrices) != len(joint_ids):
                    self.errors.append(
                        f"skins[{index}] inverseBindMatrices count {len(raw_matrices)} != joints {len(joint_ids)}"
                    )
                matrices = [tuple(raw_matrices[i]) for i in range(min(len(raw_matrices), len(joint_ids)))]
            while len(matrices) < len(joint_ids):
                matrices.append(Identity)
            skin = Skin(name, tuple(joint_ids), tuple(matrices))
            self.skin_specs[index] = (tuple(joint_ids), tuple(matrices))
            result.append(skin)
        return result

    def _attach_skins(self, nodes: list[Node], meshes: list[Mesh], skins: list[Skin]) -> list[Node]:
        source_nodes = self.doc.get("nodes", [])
        node_skin_counts: dict[int, int] = {}
        for node_index, item in enumerate(source_nodes):
            if not isinstance(item, dict) or "skin" not in item:
                continue
            try:
                skin_index = int(item["skin"])
                spec = self.skin_specs[skin_index]
            except (TypeError, ValueError, KeyError):
                self.errors.append(f"nodes[{node_index}] references missing skin {item.get('skin')}")
                continue
            node_skin_counts[skin_index] = node_skin_counts.get(skin_index, 0) + 1
            if "mesh" not in item:
                self.errors.append(f"nodes[{node_index}] has a skin but no mesh")
                continue
            try:
                mesh = meshes[int(item["mesh"])]
            except (IndexError, TypeError, ValueError):
                continue
            geometry = next((geo for geo in self.geometries if geo.id() == mesh.geometry_id), None)
            if geometry is None or geometry.skin_joints is None:
                self.errors.append(f"nodes[{node_index}] skins a mesh without JOINTS_0/WEIGHTS_0")
                continue
            joint_count = len(spec[0])
            out_of_range = [joint for joint in geometry.skin_joints if joint >= joint_count]
            if out_of_range:
                self.errors.append(
                    f"nodes[{node_index}] joint index {out_of_range[0]} exceeds skin joints count {joint_count}"
                )
        for skin_index, count in node_skin_counts.items():
            skeleton_root = self._objects("skins")[skin_index].get("skeleton")
            if skeleton_root is not None:
                try:
                    root_index = int(skeleton_root)
                    if root_index not in self.node_ids:
                        raise KeyError(root_index)
                except (TypeError, ValueError, KeyError):
                    self.errors.append(f"skins[{skin_index}] references missing skeleton node {skeleton_root}")
        replacements: dict[str, Node] = {}
        for node, item in zip(nodes, source_nodes):
            if not isinstance(item, dict) or "skin" not in item:
                continue
            try:
                skin_id = skins[int(item["skin"])].id()
            except (TypeError, ValueError, IndexError):
                skin_id = None
            replacements[node.id()] = replace(node, skin_id=skin_id)
        if not replacements:
            return nodes

        node_by_old_id = {node.id(): node for node in nodes}

        def remap(node: Node) -> Node:
            updated = replacements.get(node.id(), node)
            if updated.children:
                new_children = []
                for child_id in updated.children:
                    child_node = node_by_old_id.get(child_id)
                    new_children.append(remap(child_node).id() if child_node is not None else child_id)
                updated = replace(updated, children=tuple(new_children))
            return updated

        updated_nodes = [remap(node) for node in nodes]
        for gltf_index, old_id in self.node_ids.items():
            old_node = node_by_old_id[old_id]
            self.node_ids[gltf_index] = remap(old_node).id()
        return updated_nodes

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
