// SPDX-License-Identifier: BSD-3-Clause
// Tensor<T, R>: the owning, dense, row-major tensor.
//
// This is the "principal type" of the framework: every other tensor-like
// type (constants, and later lazy expressions) can be evaluated into one.
//
// Copies are shallow. Two copies share one buffer, so passing tensors by
// value is cheap. To keep sharing safe, writing is allowed only through a
// handle that owns its buffer alone; otherwise a precondition fails.
#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <mdspan>
#include <memory>
#include <span>
#include <utility>

#include "xinn/shape.hpp"

namespace xinn {

template <class T, std::size_t R>
class Tensor {
public:
    using value_type = T;
    static constexpr std::size_t rank = R;

    using View    = std::mdspan<const T, std::dextents<std::size_t, R>>;
    using MutView = std::mdspan<T, std::dextents<std::size_t, R>>;

    // An empty tensor: no storage.
    Tensor() = default;

    // Zero-filled tensor of the given shape.
    explicit Tensor(const Shape<R>& shape)
        : shape_(shape), buf_(std::make_shared<T[]>(shape.count())), data_(buf_.get()) {}

    Tensor(const Shape<R>& shape, T fill) : Tensor(shape) { std::ranges::fill(mut_flat(), fill); }

    // A tensor whose elements are not initialized, for results that are about
    // to be written in full. Skipping the zero-fill saves one pass over memory,
    // and lets the writes (and the operating system's page faults) happen in
    // parallel. Reading an element before writing it gives an unspecified value.
    static Tensor uninitialized(const Shape<R>& shape) {
        auto buf = std::make_shared_for_overwrite<T[]>(shape.count());
        T* data = buf.get();
        return Tensor(std::move(buf), data, shape);
    }

    // Tensor(Shape(2, 3), {1, 2, 3, 4, 5, 6}) -- values in row-major order.
    Tensor(const Shape<R>& shape, std::initializer_list<T> values)
        pre(values.size() == shape.count())
        : Tensor(shape)
    {
        std::ranges::copy(values, data_);
    }

    // Any tensor-like value of the same element type and rank converts to a
    // Tensor by evaluating it:  Matrix<float> c = a + b;
    template <class X>
        requires(!std::same_as<std::remove_cvref_t<X>, Tensor>) &&
                requires(const X& x) { { x.eval() } -> std::same_as<Tensor>; }
    Tensor(const X& x) : Tensor(x.eval()) {}

    // ---- observers ---------------------------------------------------------

    const Shape<R>& shape() const noexcept { return shape_; }
    std::size_t size() const noexcept { return shape_.count(); }
    bool empty() const noexcept { return data_ == nullptr; }

    // True if another handle (a copy or a sub-tensor) shares the buffer.
    bool shared() const noexcept { return buf_.use_count() > 1; }

    // Same buffer, same window. This is identity, not value equality; the
    // evaluator uses it to detect repeated work.
    bool identical(const Tensor& other) const noexcept {
        return data_ == other.data_ && shape_ == other.shape_;
    }

    // A tensor is already evaluated; evaluating it is a (shallow) copy.
    Tensor eval() const { return *this; }

    // ---- element access ----------------------------------------------------

    View view() const { return View(data_, shape_.dims); }
    std::span<const T> flat() const { return {data_, size()}; }

    MutView mut() pre(!empty() && !shared()) { return raw_view(); }
    std::span<T> mut_flat() pre(!empty() && !shared()) { return {data_, size()}; }

    // t(i, j) reads one element.
    template <std::convertible_to<std::size_t>... I>
        requires(sizeof...(I) == R)
    T operator()(I... idx) const {
        // Checked in the body, not with pre(): GCC 16.1 crashes (ICE) on
        // pre/post of a variadic template called with an empty pack -- which
        // is exactly the rank-0 call s().
        contract_assert(!empty() && in_bounds(idx...));
        return view()[static_cast<std::size_t>(idx)...];
    }

    // t.set(i, j, value) writes one element. The value comes last, after the
    // R indices; pack indexing (C++26) picks the pieces apart.
    template <class... A>
        requires(sizeof...(A) == R + 1)
    void set(A... args) {
        contract_assert(!empty() && !shared());  // not pre(): see operator()
        [&]<std::size_t... K>(std::index_sequence<K...>) {
            contract_assert(in_bounds(args...[K]...));
            raw_view()[static_cast<std::size_t>(args...[K])...] = static_cast<T>(args...[R]);
        }(std::make_index_sequence<R>{});
    }

    // t[i] is the (R-1)-rank slice at index i of the first dimension. It
    // shares this tensor's buffer -- no copy.
    Tensor<T, R - 1> operator[](std::size_t i) const
        requires(R > 0)
        pre(!empty() && i < shape_[0])
    {
        auto sub = [&]<std::size_t... K>(std::index_sequence<K...>) {
            return std::submdspan(raw_view(), i, ((void)K, std::full_extent)...);
        }(std::make_index_sequence<R - 1>{});

        Shape<R - 1> s;
        for (std::size_t d = 0; d < R - 1; ++d) s.dims[d] = sub.extent(d);
        return Tensor<T, R - 1>(buf_, sub.data_handle(), s);
    }

    // The same elements with another shape of the same size, sharing the
    // buffer: a (2, 6) tensor viewed as (3, 4) or (12). No copy is made.
    template <std::size_t R2>
    Tensor<T, R2> reshaped(const Shape<R2>& shape) const
        pre(shape.count() == size())
    {
        return Tensor<T, R2>(buf_, data_, shape);
    }

    // A deep copy with its own buffer.
    Tensor clone() const {
        Tensor out(shape_);
        std::ranges::copy(flat(), out.data_);
        return out;
    }

private:
    template <class, std::size_t> friend class Tensor;

    // Sub-tensor constructor: a window into someone else's buffer.
    Tensor(std::shared_ptr<T[]> buf, T* data, const Shape<R>& shape)
        : shape_(shape), buf_(std::move(buf)), data_(data) {}

    MutView raw_view() const { return MutView(data_, shape_.dims); }

    bool in_bounds(auto... idx) const {
        const std::array<std::size_t, R> at{static_cast<std::size_t>(idx)...};
        for (std::size_t d = 0; d < R; ++d)
            if (at[d] >= shape_.dims[d]) return false;
        return true;
    }

    Shape<R> shape_{};
    std::shared_ptr<T[]> buf_;
    T* data_ = nullptr;
};

template <class X> inline constexpr bool is_tensor_v = false;
template <class T, std::size_t R> inline constexpr bool is_tensor_v<Tensor<T, R>> = true;

template <class T> using Scalar = Tensor<T, 0>;
template <class T> using Vector = Tensor<T, 1>;
template <class T> using Matrix = Tensor<T, 2>;

}  // namespace xinn
