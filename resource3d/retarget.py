from __future__ import annotations

import struct

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SOF_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
}


def image_dimensions(mime_type: str, payload: bytes) -> tuple[int, int]:
    if mime_type == "image/png":
        return _png_dimensions(payload)
    if mime_type == "image/jpeg":
        return _jpeg_dimensions(payload)
    raise ValueError(f"unsupported image MIME type {mime_type!r}")


def _png_dimensions(payload: bytes) -> tuple[int, int]:
    if len(payload) < 24 or not payload.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG image")
    if payload[12:16] != b"IHDR":
        raise ValueError("PNG missing IHDR chunk")
    width, height = struct.unpack(">II", payload[16:24])
    if width <= 0 or height <= 0:
        raise ValueError("PNG has non-positive dimensions")
    return width, height


def _jpeg_dimensions(payload: bytes) -> tuple[int, int]:
    if len(payload) < 4 or payload[0:2] != b"\xff\xd8":
        raise ValueError("not a JPEG image")
    offset = 2
    while offset + 4 <= len(payload):
        if payload[offset] != 0xFF:
            offset += 1
            continue
        marker = payload[offset + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        segment_length = struct.unpack(">H", payload[offset + 2:offset + 4])[0]
        if segment_length < 2 or offset + 2 + segment_length > len(payload):
            raise ValueError("truncated JPEG segment")
        if marker in JPEG_SOF_MARKERS:
            if segment_length < 7:
                raise ValueError("truncated JPEG frame header")
            height, width = struct.unpack(">HH", payload[offset + 5:offset + 9])
            if width <= 0 or height <= 0:
                raise ValueError("JPEG has non-positive dimensions")
            return width, height
        offset += 2 + segment_length
    raise ValueError("JPEG has no frame header")
