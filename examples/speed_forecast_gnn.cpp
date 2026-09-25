// SPDX-License-Identifier: BSD-3-Clause
// Spatio-temporal speed forecasting with graph convolution.
//
// One sample is one time step at ALL detectors at once:
//
//   X (B, N, 12)  the last hour of speed at each of the N detectors
//   Y (B, N, 3)   the speed 15, 30 and 60 minutes later (3, 6, 12 intervals)
//
// Every model has the same layers -- two 64-unit ReLU layers and a linear
// output, applied at every node with shared weights -- and differs only in
// the graph Â that mixes the nodes after each hidden layer:
//
//   per-node MLP   Â = I: every detector on its own
//   GCN            Â = D^-1/2 (A + I) D^-1/2 of the detector graph
//   random graph   the same number of edges, placed at random
//   GCN + self     the same Â, plus a separate weight for each node's own
//                  features: relu(Â H W + H W_self + b)
//
// Baselines: persistence (the speed stays as it is), and the historical
// average of the detector at that time of day over the training days.
//
// Two data sets:
//   1. synthetic: the Cell Transmission Model corridor, 15 detectors in a
//      line; the graph links each detector to its neighbours on the road.
//      60 days, 48 for training.
//   2. real: PEMS08, 170 detectors, with the published detector graph
//      (295 directed edges with distances), as binary and Gaussian-kernel
//      weights.
//
// Split by time: the first days train, the last days test. Speeds are
// standardized with training statistics only. Errors are MAE in km/h, over
// all test time steps, overall and on congested targets (true speed below
// 64 km/h).
#include <chrono>
#include <cmath>
#include <fstream>
#include <print>
#include <sstream>
#include <string>
#include <vector>
#include <traffic/ctm.hpp>
#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

constexpr std::size_t history = 12;
constexpr std::size_t horizons[3] = {3, 6, 12};
constexpr std::size_t lead = 12;          // the longest horizon
constexpr float congested_below = 64.0f;  // km/h

// Speeds in km/h, (interval, detector).
struct Series {
    Matrix<float> speed;
    std::size_t per_day = 288;
    std::size_t train_days = 0, days = 0;
    std::size_t detectors() const { return speed.shape()[1]; }
};

struct Samples {
    Tensor<float, 3> x, y;           // standardized
    std::vector<std::size_t> times;  // t: the last observed interval
};

// Every `stride`-th time step t with t in [from, to) and t + lead < to.
Samples make_samples(const Series& s, std::size_t from, std::size_t to, std::size_t stride, float mean, float sd) {
    const std::size_t N = s.detectors();
    Samples out;
    for (std::size_t t = std::max(from, history - 1); t + lead < to; t += stride) out.times.push_back(t);
    const std::size_t B = out.times.size();
    out.x = Tensor<float, 3>(Shape{B, N, history});
    out.y = Tensor<float, 3>(Shape{B, N, std::size_t{3}});
    auto X = out.x.mut();
    auto Y = out.y.mut();
    auto S = s.speed.view();
    for (std::size_t b = 0; b < B; ++b) {
        const std::size_t t = out.times[b];
        for (std::size_t d = 0; d < N; ++d) {
            for (std::size_t j = 0; j < history; ++j) X[b, d, j] = (S[t + 1 - history + j, d] - mean) / sd;
            for (std::size_t h = 0; h < 3; ++h) Y[b, d, h] = (S[t + horizons[h], d] - mean) / sd;
        }
    }
    return out;
}

template <bool SelfWeight>
struct Forecaster : Module {
    GraphConv<{.activation = Activation::relu, .self_weight = SelfWeight}> g1, g2;
    NodeDense<> out;
    Forecaster(const SparseMatrix<float>& a_hat, Rng& rng)
        : g1(a_hat, history, 64, rng), g2(a_hat, 64, 64, rng), out(64, 3, rng) {}
    auto forward(const auto& x) const { return out(g2(g1(x))); }
};

// MAE per horizon, overall and on congested targets.
struct Errors {
    double all[3]{}, congested[3]{};
};

class Scorer {
public:
    explicit Scorer(const Series& s) : s_(s) {}

    // predict(t, d, h) -> km/h
    template <class F>
    Errors score(const std::vector<std::size_t>& times, F&& predict) const {
        Errors e;
        auto S = s_.speed.view();
        for (std::size_t h = 0; h < 3; ++h) {
            double sum = 0, sum_c = 0;
            std::size_t n = 0, n_c = 0;
            for (std::size_t b = 0; b < times.size(); ++b)
                for (std::size_t d = 0; d < s_.detectors(); ++d) {
                    const float truth = S[times[b] + horizons[h], d];
                    const double err = std::abs(double(predict(b, d, h)) - truth);
                    sum += err, ++n;
                    if (truth < congested_below) sum_c += err, ++n_c;
                }
            e.all[h] = sum / double(n);
            e.congested[h] = n_c ? sum_c / double(n_c) : 0;
        }
        return e;
    }

private:
    const Series& s_;
};

struct Row {
    std::string name;
    Errors e;
};

void print_table(const std::vector<Row>& rows) {
    std::println("\n{:<28}|{:>8}{:>8}{:>8} |{:>8}{:>8}{:>8}", "MAE (km/h)", "15 min", "30 min", "60 min", "15 min",
                 "30 min", "60 min");
    std::println("{:<28}|{:^24} |{:^24}", "", "all targets", "congested (< 64 km/h)");
    for (const auto& r : rows)
        std::println("{:<28}|{:>8.2f}{:>8.2f}{:>8.2f} |{:>8.2f}{:>8.2f}{:>8.2f}", r.name, r.e.all[0], r.e.all[1],
                     r.e.all[2], r.e.congested[0], r.e.congested[1], r.e.congested[2]);
}

struct Graph {
    std::string name;
    SparseMatrix<float> a_hat;
    bool self_weight = false;   // H' = relu(Â H W + H W_self + b) instead of relu(Â H W + b)
};

struct Setup {
    std::size_t train_stride = 1, epochs = 10, batch = 32, seeds = 1;
};

template <bool SelfWeight>
Errors train_and_score(const Graph& g, const Samples& train, const Samples& test, const Scorer& scorer, float mean,
                       float sd, const Setup& cfg, unsigned seed) {
    Rng rng{seed};
    Forecaster<SelfWeight> net(g.a_hat, rng);
    Adam opt(net, {.lr = 2e-3});
    const std::size_t n = train.times.size(), steps = cfg.epochs * ((n + cfg.batch - 1) / cfg.batch);
    std::size_t step = 0;
    for (std::size_t e = 0; e < cfg.epochs; ++e)
        for (auto [xb, yb] : batches(train.x, train.y, cfg.batch, rng)) {
            opt.set_learning_rate(cosine_lr(step++, steps, 2e-3, 1e-5));
            opt.zero_grad();
            backward(mse(net(xb), yb));
            opt.step();
        }
    // Predict the test set in chunks.
    const std::size_t B = test.times.size(), N = test.x.shape()[1];
    Tensor<float, 3> pred(Shape{B, N, std::size_t{3}});
    std::vector<std::size_t> idx;
    for (std::size_t start = 0; start < B; start += 256) {
        idx.clear();
        for (std::size_t i = start; i < std::min(B, start + 256); ++i) idx.push_back(i);
        const Tensor<float, 3> p = net(gather_rows(test.x, idx));
        std::ranges::copy(p.flat(), pred.mut_flat().begin() + std::ptrdiff_t(start * N * 3));
    }
    auto P = pred.view();
    return scorer.score(test.times, [&](std::size_t b, std::size_t d, std::size_t h) { return P[b, d, h] * sd + mean; });
}

std::vector<Row> run(const Series& s, const std::vector<Graph>& graphs, const Setup& cfg) {
    const std::size_t split = s.train_days * s.per_day, end = s.days * s.per_day;
    // Standardization: one mean and scale over all training speeds.
    double m = 0, v = 0;
    auto S = s.speed.view();
    for (std::size_t t = 0; t < split; ++t)
        for (std::size_t d = 0; d < s.detectors(); ++d) m += S[t, d];
    m /= double(split * s.detectors());
    for (std::size_t t = 0; t < split; ++t)
        for (std::size_t d = 0; d < s.detectors(); ++d) v += (S[t, d] - m) * (S[t, d] - m);
    const float mean = float(m), sd = float(std::sqrt(v / double(split * s.detectors())));

    const auto train = make_samples(s, 0, split, cfg.train_stride, mean, sd);
    const auto test = make_samples(s, split, end, 1, mean, sd);
    std::println("{} training time steps (days 0-{}, stride {}), {} test time steps (days {}-{}), {} detectors",
                 train.times.size(), s.train_days - 1, cfg.train_stride, test.times.size(), s.train_days, s.days - 1,
                 s.detectors());
    std::size_t congested = 0;
    for (auto t : test.times)
        for (std::size_t d = 0; d < s.detectors(); ++d) congested += S[t + lead, d] < congested_below;
    std::println("congested test targets (60 min): {:.2f}%",
                 100.0 * double(congested) / double(test.times.size() * s.detectors()));

    const Scorer scorer(s);
    std::vector<Row> rows;
    rows.push_back({"persistence", scorer.score(test.times, [&](std::size_t b, std::size_t d, std::size_t) {
                        return S[test.times[b], d];
                    })});
    // Historical average: detector and time of day, over the training days.
    Matrix<float> ha(Shape{s.per_day, s.detectors()});
    {
        auto H = ha.mut();
        for (std::size_t t = 0; t < split; ++t)
            for (std::size_t d = 0; d < s.detectors(); ++d) H[t % s.per_day, d] += S[t, d] / float(s.train_days);
    }
    auto H = ha.view();
    rows.push_back({"historical average", scorer.score(test.times, [&](std::size_t b, std::size_t d, std::size_t h) {
                        return H[(test.times[b] + horizons[h]) % s.per_day, d];
                    })});
    for (const auto& g : graphs) {
        Errors mean_e;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t k = 0; k < cfg.seeds; ++k) {
            const auto e = g.self_weight ? train_and_score<true>(g, train, test, scorer, mean, sd, cfg, unsigned(100 + k))
                                         : train_and_score<false>(g, train, test, scorer, mean, sd, cfg, unsigned(100 + k));
            for (std::size_t h = 0; h < 3; ++h) {
                mean_e.all[h] += e.all[h] / double(cfg.seeds);
                mean_e.congested[h] += e.congested[h] / double(cfg.seeds);
            }
            if (cfg.seeds > 1)
                std::println("  {:<24} seed {}: {:.3f} {:.3f} {:.3f} | {:.3f} {:.3f} {:.3f}", g.name, k, e.all[0],
                             e.all[1], e.all[2], e.congested[0], e.congested[1], e.congested[2]);
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::println("  {:<24} trained in {:.0f} s ({} run{})", g.name, secs, cfg.seeds, cfg.seeds > 1 ? "s" : "");
        rows.push_back({g.name, mean_e});
    }
    return rows;
}

// Undirected edges placed at random, as many as `count`, no self-loops or repeats.
std::vector<Edge> random_edges(std::size_t n, std::size_t count, unsigned seed) {
    Rng rng{seed};
    std::uniform_int_distribution<std::size_t> U(0, n - 1);
    std::vector<Edge> e;
    std::vector<std::vector<bool>> used(n, std::vector<bool>(n, false));
    while (e.size() < 2 * count) {
        const std::size_t i = U(rng), j = U(rng);
        if (i == j || used[i][j]) continue;
        used[i][j] = used[j][i] = true;
        e.push_back({i, j, 1.0});
        e.push_back({j, i, 1.0});
    }
    return e;
}

// How much do connected detectors tell about each other? The median, over
// pairs (i, j), of the correlation between their 15-minute speed changes on
// the training days.
double median_change_correlation(const Series& s, const std::vector<Edge>& pairs) {
    const std::size_t split = s.train_days * s.per_day;
    auto S = s.speed.view();
    std::vector<double> r;
    for (const auto& p : pairs) {
        if (p.from == p.to) continue;
        double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
        const std::size_t n = split - 3;
        for (std::size_t t = 3; t < split; ++t) {
            const double x = S[t, p.from] - S[t - 3, p.from], y = S[t, p.to] - S[t - 3, p.to];
            sx += x, sy += y, sxx += x * x, syy += y * y, sxy += x * y;
        }
        const double vx = sxx - sx * sx / double(n), vy = syy - sy * sy / double(n);
        if (vx > 0 && vy > 0) r.push_back((sxy - sx * sy / double(n)) / std::sqrt(vx * vy));
    }
    std::ranges::sort(r);
    return r.empty() ? 0 : r[r.size() / 2];
}

void print_correlations(const Series& s, const std::vector<Edge>& graph, const std::vector<Edge>& random) {
    std::println("median correlation of 15-min speed changes: graph neighbours {:.3f}, random pairs {:.3f}",
                 median_change_correlation(s, graph), median_change_correlation(s, random));
}

std::size_t undirected_pairs(const SparseMatrix<float>& sym) {
    std::size_t k = 0;
    for (const auto& e : sym.entries()) k += e.from < e.to;
    return k;
}

// ---- synthetic corridor --------------------------------------------------------------------

void corridor() {
    std::println("=== synthetic corridor (Cell Transmission Model), 15 detectors 1 km apart ===");
    traffic::Corridor c;
    c.capacity = 1800;   // veh/h/lane: a busier corridor than chapter 5's, so that queues are common
    const std::size_t n_days = 60;
    const auto days = traffic::simulate_days(c, n_days, 2026);
    const std::size_t N = c.detectors(), T = traffic::Day::intervals;
    Series s{Matrix<float>(Shape{n_days * T, N}), T, 48, n_days};
    auto m = s.speed.mut();
    for (std::size_t d = 0; d < n_days; ++d)
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t k = 0; k < N; ++k) m[d * T + t, k] = days[d].speed[t][k];

    std::vector<Edge> road;
    for (std::size_t k = 0; k + 1 < N; ++k) road.push_back({k, k + 1, 1.0}), road.push_back({k + 1, k, 1.0});
    const auto random = random_edges(N, N - 1, 5);
    print_correlations(s, road, random);
    const std::vector<Graph> graphs{
        {"per-node MLP (A = I)", SparseMatrix<float>::identity(N)},
        {"GCN, road neighbours", gcn_normalize(SparseMatrix<float>::from_edges(N, N, road))},
        {"GCN + self, road", gcn_normalize(SparseMatrix<float>::from_edges(N, N, road)), true},
        {"GCN, random graph", gcn_normalize(SparseMatrix<float>::from_edges(N, N, random))},
    };
    print_table(run(s, graphs, {.train_stride = 1, .epochs = 20, .batch = 32, .seeds = 3}));
}

// ---- PEMS08 ----------------------------------------------------------------------------------

std::vector<Edge> read_distances(const std::string& path) {
    std::ifstream in(path);
    std::vector<Edge> e;
    std::string line;
    std::getline(in, line);   // from,to,cost
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string a, b, c;
        if (std::getline(ls, a, ',') && std::getline(ls, b, ',') && std::getline(ls, c))
            e.push_back({std::stoul(a), std::stoul(b), std::stod(c)});
    }
    return e;
}

void pems08() {
    std::println("\n=== real detectors (PEMS08, Caltrans District 8), 170 detectors ===");
    auto pems = datasets::load_pems08(XINN_DATA_DIR "/pems08");
    if (!pems) return std::println("PEMS08 skipped: {}", pems.error());
    const auto dist = read_distances(XINN_DATA_DIR "/pems08/distance.csv");
    if (dist.empty()) return std::println("PEMS08 skipped: no distance.csv");
    const std::size_t N = datasets::Pems08::detectors, T = datasets::Pems08::intervals;
    Series s{Matrix<float>(Shape{T, N}), 288, 50, 62};
    auto src = pems->data.view();
    auto m = s.speed.mut();
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t d = 0; d < N; ++d) m[t, d] = src[t, d, datasets::Pems08::speed] * 1.609344f;   // mph -> km/h

    std::vector<Edge> binary = dist;
    for (auto& e : binary) e.weight = 1;
    const auto a = symmetrize(SparseMatrix<float>::from_edges(N, N, binary));
    const auto kernel = symmetrize(SparseMatrix<float>::from_edges(N, N, gaussian_kernel(dist)));
    std::println("graph: {} directed edges, {} undirected pairs; Gaussian kernel keeps {} pairs", dist.size(),
                 undirected_pairs(a), undirected_pairs(kernel));
    const auto random = random_edges(N, undirected_pairs(a), 5);
    print_correlations(s, a.entries(), random);
    const auto a_random = gcn_normalize(SparseMatrix<float>::from_edges(N, N, random));
    const std::vector<Graph> graphs{
        {"per-node MLP (A = I)", SparseMatrix<float>::identity(N)},
        {"GCN, detector graph", gcn_normalize(a)},
        {"GCN, Gaussian distances", gcn_normalize(kernel)},
        {"GCN, random graph", a_random},
        {"GCN + self, detector graph", gcn_normalize(a), true},
        {"GCN + self, random graph", a_random, true},
    };
    print_table(run(s, graphs, {.train_stride = 1, .epochs = 10, .batch = 32, .seeds = 3}));
}

}  // namespace

int main(int argc, char** argv) {
    const std::string which = argc > 1 ? argv[1] : "all";
    if (which == "all" || which == "ctm") corridor();
    if (which == "all" || which == "pems") pems08();
}
