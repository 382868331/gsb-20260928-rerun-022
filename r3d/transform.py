"""Column-major 4x4 matrix helpers matching glTF node semantics."""

from __future__ import annotations

import math
from typing import Sequence

IDENTITY = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]


def multiply(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """Return ``a * b`` for column-major 4x4 matrices."""
    result = [0.0] * 16
    for column in range(4):
        for row in range(4):
            result[column * 4 + row] = sum(
                a[k * 4 + row] * b[column * 4 + k] for k in range(4)
            )
    return result


def translation_matrix(translation: Sequence[float]) -> list[float]:
    matrix = list(IDENTITY)
    matrix[12], matrix[13], matrix[14] = translation
    return matrix


def scale_matrix(scale: Sequence[float]) -> list[float]:
    matrix = list(IDENTITY)
    matrix[0], matrix[5], matrix[10] = scale
    return matrix


def quaternion_matrix(quaternion: Sequence[float]) -> list[float]:
    # glTF quaternion: (x, y, z, w).
    x, y, z, w = quaternion
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz),       2.0 * (xz - wy),       0.0,
        2.0 * (xy - wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx),       0.0,
        2.0 * (xz + wy),       2.0 * (yz - wx),       1.0 - 2.0 * (xx + yy), 0.0,
        0.0,                   0.0,                   0.0,                   1.0,
    ]


def compose_trs(
    translation: Sequence[float] = (0.0, 0.0, 0.0),
    rotation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    scale: Sequence[float] = (1.0, 1.0, 1.0),
) -> list[float]:
    """Compose a column-major matrix: translation * rotation * scale."""
    local = multiply(quaternion_matrix(rotation), scale_matrix(scale))
    return multiply(translation_matrix(translation), local)


def is_normalized_quaternion(quaternion: Sequence[float]) -> bool:
    if len(quaternion) != 4:
        return False
    length_squared = sum(component * component for component in quaternion)
    return math.isclose(length_squared, 1.0, abs_tol=1e-5)


def transform_point(matrix: Sequence[float], point: Sequence[float]) -> list[float]:
    """Apply a 4x4 matrix to a 3D point."""
    x, y, z = point
    return [
        matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
        matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
        matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
    ]
