# SPDX-License-Identifier: BSD-3-Clause
"""Download and prepare the data sets used by the XiNN examples.

    python tools/prepare_data.py

Creates, under data/ (which git ignores):

    data/mnist/     the four MNIST IDX files (LeCun et al.), uncompressed
    data/pems08/    PEMS08 loop-detector data (Caltrans PeMS, District 8,
                    Jul-Aug 2016, 170 detectors, 5-minute intervals):
        PEMS08.npz      as published with the ASTGNN paper
        distance.csv    detector graph: from, to, cost
        pems08.f32      the same array as raw little-endian float32,
                        shape (17856, 170, 3); features: flow (veh/5 min),
                        occupancy (fraction), speed (mph)

Only the Python standard library is used.
"""
import array
import ast
import gzip
import pathlib
import shutil
import struct
import sys
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent / "data"

MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
MNIST_FILES = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]
PEMS08_BASE = "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/main/data/PEMS08/"


def fetch(url: str, dest: pathlib.Path) -> None:
    print(f"  {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def mnist() -> None:
    d = ROOT / "mnist"
    d.mkdir(parents=True, exist_ok=True)
    for name in MNIST_FILES:
        if (d / name).exists():
            continue
        gz = d / (name + ".gz")
        for base in MNIST_MIRRORS:
            try:
                fetch(base + name + ".gz", gz)
                break
            except OSError as e:
                print(f"    failed: {e}")
        else:
            sys.exit(f"could not download {name}")
        with gzip.open(gz) as src, open(d / name, "wb") as dst:
            shutil.copyfileobj(src, dst)
        gz.unlink()


def pems08() -> None:
    d = ROOT / "pems08"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "PEMS08.npz").exists():
        fetch(PEMS08_BASE + "PEMS08.npz", d / "PEMS08.npz")
    if not (d / "distance.csv").exists():
        fetch(PEMS08_BASE + "PEMS08.csv", d / "distance.csv")
    out = d / "pems08.f32"
    if out.exists():
        return
    # An .npz is a zip of .npy files; an .npy is a small header plus raw data.
    with zipfile.ZipFile(d / "PEMS08.npz") as z, z.open("data.npy") as f:
        assert f.read(6) == b"\x93NUMPY"
        major = f.read(2)[0]
        hlen = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(f.read(hlen).decode("latin1"))
        assert header["descr"] == "<f8" and not header["fortran_order"]
        assert header["shape"] == (17856, 170, 3), header["shape"]
        values = array.array("d")
        values.frombytes(f.read())
    array.array("f", values).tofile(open(out, "wb"))
    print(f"  wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    print("MNIST")
    mnist()
    print("PEMS08")
    pems08()
    print("done")
