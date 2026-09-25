# SPDX-License-Identifier: BSD-3-Clause
"""Recompress PNG files in place with full zlib compression.

XiNN Lab writes screenshots with uncompressed ("stored") zlib blocks, which
keeps its PNG writer tiny. This script rewrites them compressed:

    python tools/compress_png.py book/images/*.png

Only the Python standard library is used.
"""
import struct
import sys
import zlib


def chunks(data: bytes):
    pos = 8
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        yield kind, data[pos + 8 : pos + 8 + length]
        pos += 12 + length


def chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def compress(path: str) -> None:
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path}: not a PNG"
    parts = list(chunks(data))
    idat = b"".join(body for kind, body in parts if kind == b"IDAT")
    packed = zlib.compress(zlib.decompress(idat), 9)
    out = data[:8]
    for kind, body in parts:
        if kind == b"IDAT":
            if packed is not None:
                out += chunk(b"IDAT", packed)
                packed = None
        else:
            out += chunk(kind, body)
    open(path, "wb").write(out)
    print(f"{path}: {len(data) // 1024} KB -> {len(out) // 1024} KB")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        compress(p)
