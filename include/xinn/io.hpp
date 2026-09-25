// SPDX-License-Identifier: BSD-3-Clause
// Saving and loading parameters in the safetensors format.
//
//   save(net, "net.safetensors");
//   if (auto r = load(net, "net.safetensors"); !r) std::println("{}", r.error());
//
// safetensors (used by Hugging Face and PyTorch) is deliberately simple:
//
//   8 bytes     N, the header length, unsigned little-endian
//   N bytes     a JSON object: for each tensor
//                 "name": {"dtype": "F32", "shape": [2, 8], "data_offsets": [begin, end]}
//               plus an optional "__metadata__" object of strings
//   the rest    the raw tensor bytes, row-major; offsets count from here
//
// Names come from Module::for_each_param ("fc1.weight"), i.e. from reflection.
#pragma once

#include <bit>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "xinn/module.hpp"
#include "xinn/tensor.hpp"

namespace xinn {

static_assert(std::endian::native == std::endian::little, "safetensors stores little-endian data");

namespace detail {

template <class T>
consteval std::string_view dtype_name() {
    if constexpr (std::same_as<T, float>) return "F32";
    else if constexpr (std::same_as<T, double>) return "F64";
    else if constexpr (std::same_as<T, std::int32_t>) return "I32";
    else if constexpr (std::same_as<T, std::int64_t>) return "I64";
    else static_assert(false, "no safetensors dtype for this element type");
}

inline std::string json_string(std::string_view s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out + '"';
}

// ---- a small JSON reader: enough for safetensors headers --------------------

struct Json;
using JsonObject = std::vector<std::pair<std::string, Json>>;
using JsonArray = std::vector<Json>;

struct Json {
    // Numbers are kept as their text and converted on use.
    struct Number { std::string text; };
    std::variant<std::nullptr_t, bool, Number, std::string, JsonArray, JsonObject> value;

    const Json* find(std::string_view key) const {
        if (auto* obj = std::get_if<JsonObject>(&value))
            for (const auto& [k, v] : *obj)
                if (k == key) return &v;
        return nullptr;
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string_view text) : s_(text) {}

    std::expected<Json, std::string> parse() {
        auto v = value();
        skip_ws();
        if (v && pos_ != s_.size()) return std::unexpected(error("trailing characters"));
        return v;
    }

private:
    std::expected<Json, std::string> value() {
        skip_ws();
        if (pos_ >= s_.size()) return std::unexpected(error("unexpected end"));
        const char c = s_[pos_];
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') {
            auto str = string();
            if (!str) return std::unexpected(str.error());
            return Json{std::move(*str)};
        }
        if (literal("true")) return Json{true};
        if (literal("false")) return Json{false};
        if (literal("null")) return Json{nullptr};
        return number();
    }

    std::expected<Json, std::string> object() {
        ++pos_;
        JsonObject obj;
        skip_ws();
        if (peek('}')) return ++pos_, Json{std::move(obj)};
        while (true) {
            skip_ws();
            auto key = string();
            if (!key) return std::unexpected(key.error());
            skip_ws();
            if (!peek(':')) return std::unexpected(error("expected ':'"));
            ++pos_;
            auto v = value();
            if (!v) return v;
            obj.emplace_back(std::move(*key), std::move(*v));
            skip_ws();
            if (peek(',')) { ++pos_; continue; }
            if (peek('}')) { ++pos_; return Json{std::move(obj)}; }
            return std::unexpected(error("expected ',' or '}'"));
        }
    }

    std::expected<Json, std::string> array() {
        ++pos_;
        JsonArray arr;
        skip_ws();
        if (peek(']')) return ++pos_, Json{std::move(arr)};
        while (true) {
            auto v = value();
            if (!v) return v;
            arr.push_back(std::move(*v));
            skip_ws();
            if (peek(',')) { ++pos_; continue; }
            if (peek(']')) { ++pos_; return Json{std::move(arr)}; }
            return std::unexpected(error("expected ',' or ']'"));
        }
    }

    std::expected<std::string, std::string> string() {
        if (!peek('"')) return std::unexpected(error("expected a string"));
        ++pos_;
        std::string out;
        while (pos_ < s_.size() && s_[pos_] != '"') {
            char c = s_[pos_++];
            if (c == '\\' && pos_ < s_.size()) {
                const char e = s_[pos_++];
                switch (e) {
                    case 'n': c = '\n'; break;
                    case 't': c = '\t'; break;
                    case 'r': c = '\r'; break;
                    case 'b': c = '\b'; break;
                    case 'f': c = '\f'; break;
                    case 'u': pos_ += 4; c = '?'; break;   // non-ASCII escapes are not needed here
                    default: c = e;                        // \" \\ \/
                }
            }
            out += c;
        }
        if (pos_ >= s_.size()) return std::unexpected(error("unterminated string"));
        ++pos_;
        return out;
    }

    std::expected<Json, std::string> number() {
        const std::size_t start = pos_;
        while (pos_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[pos_])) || std::strchr("+-.eE", s_[pos_])))
            ++pos_;
        if (pos_ == start) return std::unexpected(error("unexpected character"));
        return Json{Json::Number{std::string(s_.substr(start, pos_ - start))}};
    }

    bool literal(std::string_view word) {
        if (s_.substr(pos_, word.size()) != word) return false;
        pos_ += word.size();
        return true;
    }
    bool peek(char c) const { return pos_ < s_.size() && s_[pos_] == c; }
    void skip_ws() {
        while (pos_ < s_.size() && std::strchr(" \t\r\n", s_[pos_])) ++pos_;
    }
    std::string error(std::string_view what) const { return std::format("JSON header, offset {}: {}", pos_, what); }

    std::string_view s_;
    std::size_t pos_ = 0;
};

inline std::expected<std::uint64_t, std::string> to_u64(const Json& j) {
    const auto* n = std::get_if<Json::Number>(&j.value);
    if (!n) return std::unexpected("expected a number");
    std::uint64_t v{};
    auto [end, ec] = std::from_chars(n->text.data(), n->text.data() + n->text.size(), v);
    if (ec != std::errc{} || end != n->text.data() + n->text.size()) return std::unexpected("expected an unsigned integer");
    return v;
}

}  // namespace detail

// Write every parameter of `m` to `path`.
template <ModuleType M>
std::expected<void, std::string> save(const M& m, const std::filesystem::path& path) {
    std::string header = "{";
    std::vector<std::pair<const std::byte*, std::size_t>> blobs;
    std::size_t offset = 0;

    m.for_each_param([&](const std::string& name, const auto& p) {
        using T = typename std::remove_cvref_t<decltype(p)>::value_type;
        const auto data = p.tensor().flat();
        const std::size_t bytes = data.size_bytes();

        std::string shape = "[";
        for (std::size_t d = 0; d < p.shape().rank; ++d) shape += std::format("{}{}", d ? "," : "", p.shape()[d]);
        header += std::format("{}:{{\"dtype\":\"{}\",\"shape\":{}],\"data_offsets\":[{},{}]}},",
                              detail::json_string(name), detail::dtype_name<T>(), shape, offset, offset + bytes);
        blobs.emplace_back(reinterpret_cast<const std::byte*>(data.data()), bytes);
        offset += bytes;
    });
    header += R"("__metadata__":{"format":"pt","writer":"xinn"}})";
    header.resize((header.size() + 7) / 8 * 8, ' ');   // pad so the data starts 8-byte aligned

    std::ofstream out(path, std::ios::binary);
    if (!out) return std::unexpected(std::format("cannot open {} for writing", path.string()));
    const std::uint64_t n = header.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof n);
    out.write(header.data(), std::streamsize(header.size()));
    for (auto [ptr, bytes] : blobs) out.write(reinterpret_cast<const char*>(ptr), std::streamsize(bytes));
    if (!out) return std::unexpected(std::format("write to {} failed", path.string()));
    return {};
}

// Read every parameter of `m` from `path`. Each parameter must be present
// with the same dtype and shape; tensors in the file that `m` does not have
// are ignored. On error, `m` is left unchanged.
template <ModuleType M>
std::expected<void, std::string> load(M& m, const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::unexpected(std::format("cannot open {}", path.string()));
    std::vector<char> file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    std::uint64_t n = 0;
    if (file.size() < sizeof n) return std::unexpected("file too short");
    std::memcpy(&n, file.data(), sizeof n);
    if (file.size() < sizeof n + n) return std::unexpected("header runs past the end of the file");
    auto header = detail::JsonParser(std::string_view(file.data() + sizeof n, n)).parse();
    if (!header) return std::unexpected(header.error());
    const char* data = file.data() + sizeof n + n;
    const std::size_t data_size = file.size() - sizeof n - n;

    // Check everything first, then assign, so a failure changes nothing.
    std::vector<std::function<void()>> assignments;
    std::string error;
    m.for_each_param([&](const std::string& name, auto& p) {
        if (!error.empty()) return;
        using P = std::remove_cvref_t<decltype(p)>;
        using T = typename P::value_type;
        const detail::Json* entry = header->find(name);
        if (!entry) { error = std::format("tensor '{}' not in file", name); return; }

        const detail::Json* dtype = entry->find("dtype");
        const auto* dt = dtype ? std::get_if<std::string>(&dtype->value) : nullptr;
        if (!dt || *dt != detail::dtype_name<T>()) {
            error = std::format("tensor '{}': dtype {} expected", name, detail::dtype_name<T>());
            return;
        }

        const detail::Json* shape = entry->find("shape");
        const auto* dims = shape ? std::get_if<detail::JsonArray>(&shape->value) : nullptr;
        if (!dims || dims->size() != P::rank) { error = std::format("tensor '{}': rank {} expected", name, P::rank); return; }
        for (std::size_t d = 0; d < P::rank; ++d) {
            auto v = detail::to_u64((*dims)[d]);
            if (!v || *v != p.shape()[d]) {
                error = std::format("tensor '{}': dimension {} should be {}", name, d, p.shape()[d]);
                return;
            }
        }

        const detail::Json* offs = entry->find("data_offsets");
        const auto* range = offs ? std::get_if<detail::JsonArray>(&offs->value) : nullptr;
        if (!range || range->size() != 2) { error = std::format("tensor '{}': bad data_offsets", name); return; }
        const auto begin = detail::to_u64((*range)[0]);
        const auto end = detail::to_u64((*range)[1]);
        if (!begin || !end || *end < *begin || *end > data_size || *end - *begin != p.shape().count() * sizeof(T)) {
            error = std::format("tensor '{}': bad data_offsets", name);
            return;
        }

        Tensor<T, P::rank> t(p.shape());
        std::memcpy(t.mut_flat().data(), data + *begin, *end - *begin);
        assignments.push_back([&p, t] { p.assign(t); });
    });
    if (!error.empty()) return std::unexpected(std::move(error));
    for (auto& assign : assignments) assign();
    return {};
}

}  // namespace xinn
