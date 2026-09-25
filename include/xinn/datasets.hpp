// SPDX-License-Identifier: BSD-3-Clause
// Loaders for the data sets used by the examples. Fetch the files first:
//
//   python tools/prepare_data.py
//
// Both loaders return std::expected: a missing or malformed file is an
// error value with a message, not an exception.
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <vector>

#include "xinn/tensor.hpp"

namespace xinn::datasets {

// ---- MNIST -------------------------------------------------------------------------------
//
// 28x28 grey-scale images of handwritten digits, 60,000 for training and
// 10,000 for testing (LeCun, Cortes, Burges). IDX format: big-endian 32-bit
// header fields, then one unsigned byte per pixel or label.

struct Mnist {
    Matrix<float> images;              // (n, 784), pixels scaled to [0, 1]
    std::vector<std::size_t> labels;   // 0..9
};

namespace detail {

inline std::expected<std::uint32_t, std::string> read_be32(std::ifstream& in) {
    std::uint32_t v = 0;
    if (!in.read(reinterpret_cast<char*>(&v), sizeof v)) return std::unexpected("unexpected end of file");
    if constexpr (std::endian::native == std::endian::little) v = std::byteswap(v);   // C++23
    return v;
}

}  // namespace detail

inline std::expected<Mnist, std::string> load_mnist(const std::filesystem::path& dir, bool train) {
    const auto images_path = dir / (train ? "train-images-idx3-ubyte" : "t10k-images-idx3-ubyte");
    const auto labels_path = dir / (train ? "train-labels-idx1-ubyte" : "t10k-labels-idx1-ubyte");

    std::ifstream img(images_path, std::ios::binary), lab(labels_path, std::ios::binary);
    if (!img) return std::unexpected(std::format("cannot open {} (run tools/prepare_data.py)", images_path.string()));
    if (!lab) return std::unexpected(std::format("cannot open {}", labels_path.string()));

    auto magic_i = detail::read_be32(img), n_i = detail::read_be32(img), rows = detail::read_be32(img),
         cols = detail::read_be32(img);
    auto magic_l = detail::read_be32(lab), n_l = detail::read_be32(lab);
    if (!magic_i || !n_i || !rows || !cols || !magic_l || !n_l) return std::unexpected("truncated MNIST header");
    if (*magic_i != 2051 || *magic_l != 2049) return std::unexpected("not an MNIST IDX file");
    if (*n_i != *n_l) return std::unexpected("image and label counts differ");

    const std::size_t n = *n_i, pixels = std::size_t(*rows) * *cols;
    std::vector<unsigned char> raw(n * pixels), raw_labels(n);
    if (!img.read(reinterpret_cast<char*>(raw.data()), std::streamsize(raw.size())) ||
        !lab.read(reinterpret_cast<char*>(raw_labels.data()), std::streamsize(n)))
        return std::unexpected("truncated MNIST data");

    Mnist out{Matrix<float>(Shape{n, pixels}), std::vector<std::size_t>(raw_labels.begin(), raw_labels.end())};
    auto dst = out.images.mut_flat();
    for (std::size_t i = 0; i < raw.size(); ++i) dst[i] = float(raw[i]) / 255.0f;
    return out;
}

// ---- PEMS08 ------------------------------------------------------------------------------
//
// Loop-detector data from Caltrans PeMS, District 8 (San Bernardino),
// July-August 2016: 170 detectors, 17,856 five-minute intervals (62 days).
// As published with the ASTGNN / ASTGCN papers; converted to raw float32 by
// tools/prepare_data.py.

struct Pems08 {
    static constexpr std::size_t intervals = 17856, detectors = 170, per_day = 288;
    enum Feature : std::size_t { flow = 0, occupancy = 1, speed = 2 };
    // (interval, detector, feature): flow in veh/5 min over all lanes,
    // occupancy as a fraction of time, speed in mph.
    Tensor<float, 3> data;
};

inline std::expected<Pems08, std::string> load_pems08(const std::filesystem::path& dir) {
    const auto path = dir / "pems08.f32";
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::unexpected(std::format("cannot open {} (run tools/prepare_data.py)", path.string()));
    Pems08 out{Tensor<float, 3>(Shape{Pems08::intervals, Pems08::detectors, std::size_t{3}})};
    auto dst = out.data.mut_flat();
    if (!in.read(reinterpret_cast<char*>(dst.data()), std::streamsize(dst.size_bytes())))
        return std::unexpected(std::format("{} is shorter than expected", path.string()));
    return out;
}

}  // namespace xinn::datasets
