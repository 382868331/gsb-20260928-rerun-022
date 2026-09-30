from __future__ import annotations

import math
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


def inverse(matrix: Matrix4) -> Matrix4:
    values = [list(matrix[row * 4:row * 4 + 4]) for row in range(4)]
    identity = [[float(row == column) for column in range(4)] for row in range(4)]
    for column in range(4):
        pivot = max(range(column, 4), key=lambda row: abs(values[row][column]))
        if abs(values[pivot][column]) < 1e-12:
            raise ValueError("matrix is not invertible")
        if pivot != column:
            values[column], values[pivot] = values[pivot], values[column]
            identity[column], identity[pivot] = identity[pivot], identity[column]
        pivot_value = values[column][column]
        values[column] = [value / pivot_value for value in values[column]]
        identity[column] = [value / pivot_value for value in identity[column]]
        for row in range(4):
            if row == column:
                continue
            factor = values[row][column]
            values[row] = [a - factor * b for a, b in zip(values[row], values[column])]
            identity[row] = [a - factor * b for a, b in zip(identity[row], identity[column])]
    return tuple(value for row in identity for value in row)


def skin_matrix(
    joints: tuple[int, ...],
    weights: tuple[float, ...],
    joint_world: list[Matrix4],
    inverse_bind: list[Matrix4],
) -> Matrix4:
    blended = [0.0] * 16
    for joint, weight in zip(joints, weights):
        if weight == 0.0:
            continue
        palette = multiply(joint_world[joint], inverse_bind[joint])
        for index, value in enumerate(palette):
            blended[index] += weight * value
    return tuple(blended)


def skin_point(
    point: Vector3,
    joints: tuple[int, ...],
    weights: tuple[float, ...],
    joint_world: list[Matrix4],
    inverse_bind: list[Matrix4],
) -> Vector3:
    return transform_point(
        skin_matrix(joints, weights, joint_world, inverse_bind),
        point,
    )
