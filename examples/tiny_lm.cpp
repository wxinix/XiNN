// SPDX-License-Identifier: BSD-3-Clause
// A tiny character-level language model, trained on this book.
//
// All book/*.md files are read as one text of Unicode characters. Every
// tenth block of 1,024 characters is held out for validation. A small causal
// transformer learns to predict the next character from the 64 before it.
// The loss is reported in nats and in bits per character; a model that knew
// nothing but the character frequencies would need about 4.9 bits.
//
//   tiny_lm [steps] [d_model]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <numbers>
#include <print>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

constexpr std::size_t context = 64, layers = 2, heads = 4, batch = 32;

// ---- text -----------------------------------------------------------------------------------

std::u32string decode_utf8(const std::string& s) {
    std::u32string out;
    for (std::size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i]);
        const std::size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
        char32_t cp = n == 1 ? c : c & (0x3F >> (n - 1));
        for (std::size_t k = 1; k < n && i + k < s.size(); ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        out.push_back(cp);
        i += n;
    }
    return out;
}

std::string encode_utf8(char32_t cp) {
    std::string out;
    if (cp < 0x80) out += char(cp);
    else if (cp < 0x800) out += {char(0xC0 | (cp >> 6)), char(0x80 | (cp & 0x3F))};
    else if (cp < 0x10000) out += {char(0xE0 | (cp >> 12)), char(0x80 | ((cp >> 6) & 0x3F)), char(0x80 | (cp & 0x3F))};
    else out += {char(0xF0 | (cp >> 18)), char(0x80 | ((cp >> 12) & 0x3F)), char(0x80 | ((cp >> 6) & 0x3F)), char(0x80 | (cp & 0x3F))};
    return out;
}

struct Corpus {
    std::vector<char32_t> chars;               // id -> character
    std::map<char32_t, std::size_t> id;         // character -> id
    std::vector<std::size_t> train, valid;      // the text as ids
    std::size_t files = 0;
};

Corpus load_book(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> paths;
    for (const auto& e : std::filesystem::directory_iterator(dir))
        if (e.path().extension() == ".md") paths.push_back(e.path());
    std::ranges::sort(paths);
    std::u32string text;
    for (const auto& p : paths) {
        std::ifstream in(p, std::ios::binary);
        std::stringstream ss;
        ss << in.rdbuf();
        text += decode_utf8(ss.str());
    }
    Corpus c;
    c.files = paths.size();
    for (char32_t ch : text) c.id.try_emplace(ch, 0);
    for (auto& [ch, i] : c.id) {
        i = c.chars.size();
        c.chars.push_back(ch);
    }
    for (std::size_t i = 0; i < text.size(); ++i)
        ((i / 1024) % 10 == 9 ? c.valid : c.train).push_back(c.id[text[i]]);
    return c;
}

// ---- model ----------------------------------------------------------------------------------

struct CharLm : Module {
    Embedding<> embed;
    TransformerBlock<> block1, block2;
    LayerNorm<> norm;
    TokenDense<> head;

    CharLm(std::size_t vocab, std::size_t d_model, Rng& rng)
        : embed(vocab, d_model, rng),
          block1({.d_model = d_model, .heads = heads, .causal = true}, rng),
          block2({.d_model = d_model, .heads = heads, .causal = true}, rng),
          norm(d_model), head(d_model, vocab, rng) {}

    // (B, T) ids -> (B, T, vocab) logits for the next character.
    auto forward(const Tensor<std::size_t, 2>& ids) const {
        const auto pe = sinusoidal_positions<float>(ids.shape()[1], embed.table.shape()[1]);
        return head(norm(block2(block1(embed(ids) + pe))));
    }
};
static_assert(layers == 2, "CharLm spells out its two blocks");

// Windows of context + 1 ids starting at `starts`: inputs and one-hot next characters.
std::pair<Tensor<std::size_t, 2>, Matrix<float>> make_batch(const std::vector<std::size_t>& text,
                                                            const std::vector<std::size_t>& starts,
                                                            std::size_t vocab) {
    const std::size_t B = starts.size();
    Tensor<std::size_t, 2> x(Shape{B, context});
    Matrix<float> y(Shape{B * context, vocab});
    for (std::size_t b = 0; b < B; ++b)
        for (std::size_t t = 0; t < context; ++t) {
            x.set(b, t, text[starts[b] + t]);
            y.set(b * context + t, text[starts[b] + t + 1], 1.0f);
        }
    return {x, y};
}

float loss_of(const CharLm& lm, const Tensor<std::size_t, 2>& x, const Matrix<float>& y) {
    const std::size_t rows = y.shape()[0], vocab = y.shape()[1];
    return Scalar<float>(softmax_cross_entropy(reshape(lm(x), Shape{rows, vocab}), y))();
}

// Mean loss over non-overlapping windows of the whole validation text.
double validation_loss(const CharLm& lm, const Corpus& c) {
    std::vector<std::size_t> starts;
    for (std::size_t s = 0; s + context + 1 <= c.valid.size(); s += context) starts.push_back(s);
    double total = 0;
    for (std::size_t i = 0; i < starts.size(); i += 64) {
        std::vector<std::size_t> part(starts.begin() + long(i), starts.begin() + long(std::min(i + 64, starts.size())));
        auto [x, y] = make_batch(c.valid, part, c.chars.size());
        total += double(loss_of(lm, x, y)) * double(part.size());
    }
    return total / double(starts.size());
}

std::string generate(const CharLm& lm, const Corpus& c, const std::string& prompt, std::size_t n, float temperature,
                     Rng& rng) {
    std::vector<std::size_t> ids;
    for (char32_t ch : decode_utf8(prompt)) ids.push_back(c.id.at(ch));
    std::string out = prompt;
    const std::size_t V = c.chars.size();
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t len = std::min(ids.size(), context);
        Tensor<std::size_t, 2> x(Shape{1uz, len});
        for (std::size_t t = 0; t < len; ++t) x.set(0, t, ids[ids.size() - len + t]);
        const Tensor<float, 3> logits = lm(x);
        std::vector<double> p(V);
        double top = -1e30;
        for (std::size_t v = 0; v < V; ++v) top = std::max(top, double(logits(0, len - 1, v)));
        for (std::size_t v = 0; v < V; ++v) p[v] = std::exp((double(logits(0, len - 1, v)) - top) / temperature);
        const std::size_t next = std::discrete_distribution<std::size_t>(p.begin(), p.end())(rng);
        ids.push_back(next);
        out += encode_utf8(c.chars[next]);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t steps = argc > 1 ? std::stoul(argv[1]) : 2000;
    const std::size_t d_model = argc > 2 ? std::stoul(argv[2]) : 64;
    const Corpus corpus = load_book(XINN_BOOK_DIR);
    const std::size_t V = corpus.chars.size();
    std::println("book: {} files, {} characters, {} distinct; {} for training, {} for validation", corpus.files,
                 corpus.train.size() + corpus.valid.size(), V, corpus.train.size(), corpus.valid.size());

    // Unigram baseline: the entropy of the training character frequencies, on the validation text.
    {
        std::vector<double> freq(V, 1.0);   // add-one smoothing
        for (auto i : corpus.train) freq[i] += 1;
        const double total = double(corpus.train.size() + V);
        double nats = 0;
        for (auto i : corpus.valid) nats -= std::log(freq[i] / total);
        nats /= double(corpus.valid.size());
        std::println("unigram baseline: validation {:.3f} nats = {:.3f} bits/char", nats, nats / std::numbers::ln2);
    }

    Rng rng{2026};
    CharLm lm(V, d_model, rng);
    std::println("model: {} layers, d_model {}, {} heads, context {}; {} parameters", layers, d_model, heads, context,
                 lm.parameter_count());

    const double lr0 = 3e-3;
    Adam opt(lm, {.lr = lr0, .weight_decay = 0.01});
    std::uniform_int_distribution<std::size_t> pick(0, corpus.train.size() - context - 2);
    const auto t0 = std::chrono::steady_clock::now();
    double running = 0, best = 1e9;
    std::size_t best_step = 0;
    for (std::size_t step = 1; step <= steps; ++step) {
        const double warm = std::min(1.0, double(step) / 100);
        opt.set_learning_rate(warm * cosine_lr(step, steps, lr0, lr0 / 20));
        std::vector<std::size_t> starts(batch);
        for (auto& s : starts) s = pick(rng);
        auto [x, y] = make_batch(corpus.train, starts, V);
        opt.zero_grad();
        const float loss = backward(softmax_cross_entropy(reshape(lm(x), Shape{batch * context, V}), y));
        opt.step();
        running = step == 1 ? loss : 0.98 * running + 0.02 * loss;
        if (step % 250 == 0 || step == steps) {
            const double val = validation_loss(lm, corpus);
            if (val < best) best = val, best_step = step;
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::println("step {:>5}  train {:.3f} nats ({:.2f} bits/char)  validation {:.3f} nats ({:.2f} bits/char)  {:.0f} s",
                         step, running, running / std::numbers::ln2, val, val / std::numbers::ln2, sec);
        }
    }
    std::println("best validation: {:.3f} nats = {:.2f} bits/char (step {})", best, best / std::numbers::ln2, best_step);

    Rng sampler{7};
    for (const char* prompt : {"The gradient ", "A tensor ", "Traffic "})
        for (float temp : {0.5f, 0.8f}) {
            std::println("\n--- prompt \"{}\", temperature {} ---", prompt, temp);
            std::println("{}", generate(lm, corpus, prompt, 240, temp, sampler));
        }
}
