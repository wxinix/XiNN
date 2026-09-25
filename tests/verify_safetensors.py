# SPDX-License-Identifier: BSD-3-Clause
"""Check a file written by xinn::save against the safetensors format.

Uses only the Python standard library, following the format description at
https://github.com/huggingface/safetensors. If the `safetensors` package is
installed, the file is also loaded with it.

Usage: python verify_safetensors.py FILE
"""
import json
import struct
import sys

DTYPE_SIZE = {"F64": 8, "F32": 4, "I64": 8, "I32": 4}


def main(path: str) -> int:
    data = open(path, "rb").read()
    (n,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + n].decode("utf-8"))
    body = data[8 + n :]

    assert n % 8 == 0, "header length should keep data 8-byte aligned"
    meta = header.pop("__metadata__", {})
    assert all(isinstance(v, str) for v in meta.values()), "metadata values must be strings"

    spans = []
    for name, info in header.items():
        count = 1
        for d in info["shape"]:
            count *= d
        begin, end = info["data_offsets"]
        assert end - begin == count * DTYPE_SIZE[info["dtype"]], f"{name}: size mismatch"
        spans.append((begin, end))

    spans.sort()
    position = 0
    for begin, end in spans:  # tensors must tile the byte buffer exactly
        assert begin == position, "gap or overlap in data_offsets"
        position = end
    assert position == len(body), "trailing bytes after the last tensor"

    try:
        from safetensors.numpy import load_file  # optional
    except ImportError:
        print(f"ok: {len(header)} tensors (safetensors package not installed; format checked by hand)")
        return 0
    tensors = load_file(path)
    assert set(tensors) == set(header)
    print(f"ok: {len(header)} tensors, also loaded by the safetensors package")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
