"""r3d: format-neutral 3D resource import service.

The internal resource model is content addressed and knows nothing about
glTF or any other source format. New format importers produce the same
resource kinds and ids defined in :mod:`r3d.model`.
"""

from .model import (
    BLOB_PREFIX,
    GEOMETRY,
    IMAGE,
    MATERIAL,
    NODE,
    SCENE,
    ResourceError,
    canonical_json,
    make_geometry,
    make_image,
    make_material,
    make_node,
    make_scene,
    resource_id,
)

__all__ = [
    "GEOMETRY",
    "MATERIAL",
    "IMAGE",
    "NODE",
    "SCENE",
    "BLOB_PREFIX",
    "ResourceError",
    "canonical_json",
    "resource_id",
    "make_geometry",
    "make_material",
    "make_image",
    "make_node",
    "make_scene",
]
