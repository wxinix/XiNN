// SPDX-License-Identifier: BSD-3-Clause
// A minimal PNG writer for screenshots: 8-bit RGBA, zlib "stored" (not
// compressed) blocks. Larger than a compressed PNG, but short, and any image
// viewer opens it.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string_view>
#include <vector>

namespace lab {

namespace detail {

inline std::uint32_t crc32(std::span<const std::uint8_t> data, std::uint32_t crc = 0xffffffffu) {
    static const auto table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t n = 0; n < 256; ++n) {
            std::uint32_t c = n;
            for (int k = 0; k < 8; ++k) c = c & 1 ? 0xedb88320u ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }();
    for (auto b : data) crc = table[(crc ^ b) & 0xff] ^ (crc >> 8);
    return crc;
}

inline void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int s = 24; s >= 0; s -= 8) out.push_back(std::uint8_t(v >> s));
}

inline void chunk(std::ofstream& f, std::string_view type, const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> buf;
    put32(buf, std::uint32_t(data.size()));
    buf.insert(buf.end(), type.begin(), type.end());
    buf.insert(buf.end(), data.begin(), data.end());
    const auto crc = crc32(std::span(buf).subspan(4)) ^ 0xffffffffu;
    put32(buf, crc);
    f.write(reinterpret_cast<const char*>(buf.data()), std::streamsize(buf.size()));
}

}  // namespace detail

// rgba: width * height * 4 bytes, top row first.
inline bool write_png(const std::filesystem::path& path, int width, int height, std::span<const std::uint8_t> rgba) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const std::uint8_t signature[] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    f.write(reinterpret_cast<const char*>(signature), sizeof signature);

    std::vector<std::uint8_t> ihdr;
    detail::put32(ihdr, std::uint32_t(width));
    detail::put32(ihdr, std::uint32_t(height));
    ihdr.insert(ihdr.end(), {8, 6, 0, 0, 0});   // 8 bits, RGBA, deflate, no filter, no interlace
    detail::chunk(f, "IHDR", ihdr);

    // Raw scanlines, each prefixed with filter type 0.
    std::vector<std::uint8_t> raw;
    const std::size_t row = std::size_t(width) * 4;
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgba.begin() + std::ptrdiff_t(y * row), rgba.begin() + std::ptrdiff_t((y + 1) * row));
    }
    // zlib stream of stored blocks (at most 65535 bytes each), then Adler-32.
    std::vector<std::uint8_t> z{0x78, 0x01};
    std::uint32_t a = 1, b = 0;
    for (auto byte : raw) a = (a + byte) % 65521, b = (b + a) % 65521;
    for (std::size_t pos = 0; pos < raw.size(); pos += 65535) {
        const std::size_t len = std::min<std::size_t>(65535, raw.size() - pos);
        z.push_back(pos + len == raw.size() ? 1 : 0);
        z.push_back(std::uint8_t(len)), z.push_back(std::uint8_t(len >> 8));
        z.push_back(std::uint8_t(~len)), z.push_back(std::uint8_t(~len >> 8));
        z.insert(z.end(), raw.begin() + std::ptrdiff_t(pos), raw.begin() + std::ptrdiff_t(pos + len));
    }
    detail::put32(z, (b << 16) | a);
    detail::chunk(f, "IDAT", z);
    detail::chunk(f, "IEND", {});
    return bool(f);
}

}  // namespace lab
