// SPDX-License-Identifier: BSD-3-Clause
// Measuring results.
#pragma once

#include <cmath>
#include <cstddef>
#include <format>
#include <span>
#include <string>
#include <vector>

namespace xinn {

inline double accuracy(std::span<const std::size_t> predicted, std::span<const std::size_t> truth)
    pre(predicted.size() == truth.size() && !truth.empty())
{
    std::size_t ok = 0;
    for (std::size_t i = 0; i < truth.size(); ++i) ok += predicted[i] == truth[i];
    return double(ok) / double(truth.size());
}

// counts[t][p]: how many samples of true class t were predicted as p.
struct ConfusionMatrix {
    std::vector<std::vector<std::size_t>> counts;

    ConfusionMatrix(std::span<const std::size_t> predicted, std::span<const std::size_t> truth, std::size_t classes)
        pre(predicted.size() == truth.size())
        : counts(classes, std::vector<std::size_t>(classes))
    {
        for (std::size_t i = 0; i < truth.size(); ++i) ++counts[truth[i]][predicted[i]];
    }

    // Of the samples truly in class c, the share predicted as c.
    double recall(std::size_t c) const {
        std::size_t total = 0;
        for (auto n : counts[c]) total += n;
        return total ? double(counts[c][c]) / double(total) : 0.0;
    }

    // Of the samples predicted as class c, the share truly in c.
    double precision(std::size_t c) const {
        std::size_t total = 0;
        for (const auto& row : counts) total += row[c];
        return total ? double(counts[c][c]) / double(total) : 0.0;
    }

    double f1(std::size_t c) const {
        const double p = precision(c), r = recall(c);
        return p + r > 0 ? 2 * p * r / (p + r) : 0.0;
    }

    // The mean F1 over classes: unlike accuracy, a rare class counts as much
    // as a common one.
    double macro_f1() const {
        double s = 0;
        for (std::size_t c = 0; c < counts.size(); ++c) s += f1(c);
        return s / double(counts.size());
    }

    std::string table(std::span<const std::string> names) const {
        std::string out = std::format("{:>14}", "true \\ pred");
        for (const auto& n : names) out += std::format("{:>12}", n);
        out += std::format("{:>10}\n", "recall");
        for (std::size_t t = 0; t < counts.size(); ++t) {
            out += std::format("{:>14}", names[t]);
            for (auto n : counts[t]) out += std::format("{:>12}", n);
            out += std::format("{:>9.1f}%\n", 100 * recall(t));
        }
        return out;
    }
};

inline double mean_absolute_error(std::span<const float> pred, std::span<const float> truth)
    pre(pred.size() == truth.size() && !truth.empty())
{
    double s = 0;
    for (std::size_t i = 0; i < truth.size(); ++i) s += std::abs(double(pred[i]) - truth[i]);
    return s / double(truth.size());
}

// Mean absolute percentage error, in percent.
inline double mean_absolute_percentage_error(std::span<const float> pred, std::span<const float> truth)
    pre(pred.size() == truth.size() && !truth.empty())
{
    double s = 0;
    for (std::size_t i = 0; i < truth.size(); ++i) s += std::abs((double(pred[i]) - truth[i]) / truth[i]);
    return 100 * s / double(truth.size());
}

}  // namespace xinn
