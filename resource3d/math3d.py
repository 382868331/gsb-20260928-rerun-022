from __future__ import annotations

from typing import Tuple

Matrix4 = Tuple[float, ...]
Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
Identity: Matrix4 = tuple(
    1.0 if column == row else 0.0 for column in range(4) for row in range(4)
)


def multiply(a: Matrix4, b: Matrix4) -> Matrix4:
    return tuple(
        sum(a[row * 4 + k] * b[k * 4 + column] for k in range(4))
        for row in range(4)
        for column in range(4)
    )


def translation_matrix(translation: Vector3) -> Matrix4:
    x, y, z = translation
    return (1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1)


def scale_matrix(scale: Vector3) -> Matrix4:
    x, y, z = scale
    return (x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1)


def rotation_matrix(rotation: Quaternion) -> Matrix4:
    x, y, z, w = rotation
    x2, y2, z2 = x + x, y + y, z + z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    return (
        1 - (yy + zz), xy - wz, xz + wy, 0,
        xy + wz, 1 - (xx + zz), yz - wx, 0,
        xz - wy, yz + wx, 1 - (xx + yy), 0,
        0, 0, 0, 1,
    )


def trs(translation: Vector3, rotation: Quaternion, scale: Vector3) -> Matrix4:
    return multiply(
        translation_matrix(translation),
        multiply(rotation_matrix(rotation), scale_matrix(scale)),
    )


def transform_point(matrix: Matrix4, point: Vector3) -> Vector3:
    x, y, z = point
    return (
        matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3],
        matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7],
        matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11],
    )


def almost_equal_vector(a: Vector3, b: Vector3, tolerance: float = 1e-9) -> bool:
    return all(math.isclose(x, y, abs_tol=tolerance) for x, y in zip(a, b))
