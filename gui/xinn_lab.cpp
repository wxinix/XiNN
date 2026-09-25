// SPDX-License-Identifier: BSD-3-Clause
// XiNN Lab: an interactive front end for the traffic examples.
//
//   xinn_lab                      open the window
//   xinn_lab --screenshots DIR    render each page once, save PNGs, exit
//
// Tabs:
//   Car following    the 1959 model of Herman et al.: drag C across 1/e,
//                    1/2 and pi/2 and watch the line of cars; swap in a
//                    driver learned from data
//   Training data    snapshots of what the learned drivers were trained on
//   Freeway (CTM)    a simulated day on the Cell Transmission Model corridor
//
// Built with Dear ImGui and ImPlot on GLFW + OpenGL 3.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <numbers>
#include <print>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

#include <car_following/herman1959.hpp>
#include <car_following/learn.hpp>
#include <traffic/ctm.hpp>

#include "png.hpp"

namespace lab {

using namespace carfollow;

constexpr double inv_e = 1 / std::numbers::e, half_pi = std::numbers::pi / 2;

// Axis limits that show all of v, with a margin.
std::pair<double, double> limits_of(const std::vector<double>& v, double margin = 0.08) {
    double lo = v.empty() ? 0 : v[0], hi = lo;
    for (double x : v) lo = std::min(lo, x), hi = std::max(hi, x);
    const double pad = std::max(hi - lo, 1e-6) * margin;
    return {lo - pad, hi + pad};
}

// Every n-th value of a series, to keep plots light.
std::vector<double> every(const std::vector<double>& v, std::size_t n) {
    std::vector<double> out;
    for (std::size_t i = 0; i < v.size(); i += n) out.push_back(v[i]);
    return out;
}

// ---- Car following ------------------------------------------------------------------------------

struct CarFollowingPage {
    // Scenario (the paper's defaults for Figure 5).
    float C = 0.75f, delta = 1.5f, u = 70, gap = 70, duration = 60;
    int cars = 8;
    enum Driver { True, Fitted, Neural };
    int driver = True;

    // A driver learned on a background thread.
    struct Result {
        Observations obs;
        Learned learned;
        double C, delta;
    };
    std::unique_ptr<Result> result;   // the latest finished one
    std::mutex mutex;
    std::unique_ptr<Result> pending;
    std::jthread job;
    std::atomic<bool> learning = false;

    Trajectories tr;
    bool dirty = true;

    void learn_now(bool background) {
        auto task = [this, C = double(C), delta = double(delta)](std::stop_token) {
            Rng rng{1959};
            auto obs = observe(C, delta, rng);
            auto learned = learn(obs, rng);
            std::lock_guard lock(mutex);
            pending.reset(new Result{std::move(obs), std::move(learned), C, delta});
            learning = false;
        };
        learning = true;
        if (background) job = std::jthread(task);
        else task(std::stop_token{});
    }

    void poll() {
        std::lock_guard lock(mutex);
        if (pending) {
            result = std::move(pending);
            dirty = true;
        }
    }

    void simulate_now() {
        Scenario s{.delta = delta, .C = C, .u = u, .gap = gap, .duration = duration, .cars = std::size_t(cars)};
        Follower f = linear_follower(s.lambda_m());
        if (result && driver == Fitted) f = linear_follower(result->learned.lambda_m);
        if (result && driver == Neural) f = neural_follower(result->learned);
        tr = simulate(s, lead_pulse, f);
        dirty = false;
    }

    void draw() {
        poll();
        if (dirty) simulate_now();

        ImGui::BeginChild("controls", ImVec2(340, 0), ImGuiChildFlags_Borders);
        ImGui::SeparatorText("Scenario");
        dirty |= ImGui::SliderFloat("C", &C, 0.1f, 2.0f, "%.3f");
        ImGui::SetItemTooltip("C = lambda Delta / M, the sensitivity times the reaction time");
        dirty |= ImGui::SliderFloat("Delta (s)", &delta, 0.5f, 3.0f, "%.2f");
        dirty |= ImGui::SliderInt("cars", &cars, 2, 12);
        dirty |= ImGui::SliderFloat("duration (s)", &duration, 10, 120, "%.0f");
        dirty |= ImGui::SliderFloat("speed u (ft/s)", &u, 20, 100, "%.0f");
        dirty |= ImGui::SliderFloat("spacing (ft)", &gap, 20, 150, "%.0f");
        if (ImGui::Button("Figure 3")) C = float(inv_e), cars = 2, delta = 1.5f, u = gap = 70, duration = 20, dirty = true;
        ImGui::SameLine();
        if (ImGui::Button("Figure 5")) C = 0.75f, cars = 8, delta = 1.5f, u = gap = 70, duration = 60, dirty = true;
        ImGui::SameLine();
        if (ImGui::Button("Figure 6")) C = 0.8f, cars = 9, delta = 2.0f, u = gap = 40, duration = 30, dirty = true;

        ImGui::SeparatorText("Theory");
        const char* local = C < inv_e ? "no oscillation" : C < half_pi ? "damped oscillation" : "growing oscillation";
        const char* line = C < 0.5 ? "disturbances shrink down the line" : "disturbances grow down the line";
        ImGui::TextWrapped("C = %.3f: two cars show %s; a line of cars: %s.", C, local, line);
        ImGui::TextDisabled("thresholds: 1/e = %.3f, 1/2, pi/2 = %.3f", inv_e, half_pi);

        ImGui::SeparatorText("This run");
        if (const double t = first_collision(tr); t > 0) {
            ImGui::TextColored(ImVec4(1, 0.35f, 0.3f, 1), "collision at t = %.1f s", t);
        } else if (tr.cars == 2) {
            const auto st = settling(tr, 1, gap);
            ImGui::Text("%s (late/early x%.2f)", local_verdict(st), st.growth());
        } else {
            const auto d = disturbance_by_car(tr, gap);
            ImGui::Text("disturbance last/first car: x%.2f", d.back() / d.front());
        }

        ImGui::SeparatorText("Driver");
        dirty |= ImGui::RadioButton("1959 law (true driver)", &driver, True);
        ImGui::BeginDisabled(!result);
        dirty |= ImGui::RadioButton("1959 law, lambda/M learned", &driver, Fitted);
        dirty |= ImGui::RadioButton("neural network, learned", &driver, Neural);
        ImGui::EndDisabled();
        ImGui::BeginDisabled(learning);
        if (ImGui::Button(learning ? "learning..." : "Learn this driver from data")) learn_now(true);
        ImGui::EndDisabled();
        if (result)
            ImGui::TextWrapped("learned from a driver with C = %.3f, Delta = %.2f s: fitted C = %.3f, network error "
                               "%.2f ft/s^2 (noise 0.30)",
                               result->C, result->delta, result->learned.lambda_m * result->delta,
                               result->learned.net_rmse);
        ImGui::EndChild();

        ImGui::SameLine();
        ImGui::BeginChild("plots");
        const std::size_t stride = 5;
        std::vector<double> t;
        for (std::size_t i = 0; i <= tr.steps; i += stride) t.push_back(tr.time(i));
        const float h = (ImGui::GetContentRegionAvail().y - 8) / 2;
        if (ImPlot::BeginPlot("Spacing to the car ahead, minus initial spacing", ImVec2(-1, h))) {
            ImPlot::SetupAxes("time (s)", "ft", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            for (std::size_t n = 1; n < tr.cars; ++n) {
                std::vector<double> d;
                for (std::size_t i = 0; i <= tr.steps; i += stride) d.push_back(tr.spacing(n, i) - gap);
                const std::string label = std::format("{}-{}", n, n + 1);
                ImPlot::PlotLine(label.c_str(), t.data(), d.data(), int(t.size()));
            }
            ImPlot::EndPlot();
        }
        if (ImPlot::BeginPlot("Car positions relative to steady motion (x - u t)", ImVec2(-1, h))) {
            ImPlot::SetupAxes("time (s)", "ft", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            for (std::size_t n = 0; n < tr.cars; ++n) {
                std::vector<double> x;
                for (std::size_t i = 0; i <= tr.steps; i += stride) x.push_back(tr.pos(n, i) - u * tr.time(i));
                const std::string label = std::format("car {}", n + 1);
                ImPlot::PlotLine(label.c_str(), t.data(), x.data(), int(t.size()));
            }
            ImPlot::EndPlot();
        }
        ImGui::EndChild();
    }
};

// ---- Training data -----------------------------------------------------------------------------

void draw_training_data(CarFollowingPage& cf) {
    cf.poll();
    if (!cf.result) {
        ImGui::TextWrapped("No driver learned yet. On the Car following tab, press \"Learn this driver from data\".");
        return;
    }
    const auto& r = *cf.result;
    ImGui::Text("Driver with C = %.3f, Delta = %.2f s, observed in 60 episodes of a 6-car line: %zu samples "
                "(what each follower saw Delta ago -> its acceleration, noise 0.3 ft/s^2).",
                r.C, r.delta, r.obs.y.shape()[0]);

    const float h = (ImGui::GetContentRegionAvail().y - 8) / 2;
    if (ImGui::BeginTable("episodes", 2, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextColumn();
        const auto& ex = r.obs.examples[0];
        std::vector<double> t, a0;
        for (std::size_t i = 0; i <= ex.steps; i += 5) t.push_back(ex.time(i)), a0.push_back(ex.acc(0, i));
        if (ImPlot::BeginPlot("Episode 1: lead car's random manoeuvres", ImVec2(-1, h))) {
            ImPlot::SetupAxes("time (s)", "lead acceleration (ft/s^2)");
            const auto [lo, hi] = limits_of(a0);
            ImPlot::SetupAxesLimits(0, ex.time(ex.steps), lo, hi, ImPlotCond_Always);
            ImPlot::PlotLine("lead car", t.data(), a0.data(), int(t.size()));
            ImPlot::EndPlot();
        }
        ImGui::TableNextColumn();
        if (ImPlot::BeginPlot("Episode 1: speeds of the six cars", ImVec2(-1, h))) {
            ImPlot::SetupAxes("time (s)", "speed (ft/s)");
            const auto [lo, hi] = limits_of(ex.v);
            ImPlot::SetupAxesLimits(0, ex.time(ex.steps), lo, hi, ImPlotCond_Always);
            for (std::size_t n = 0; n < ex.cars; ++n) {
                std::vector<double> v;
                for (std::size_t i = 0; i <= ex.steps; i += 5) v.push_back(ex.vel(n, i));
                const std::string label = n == 0 ? std::string("lead") : std::format("car {}", n + 1);
                ImPlot::PlotLine(label.c_str(), t.data(), v.data(), int(t.size()));
            }
            ImPlot::EndPlot();
        }
        ImGui::EndTable();
    }

    // The samples: acceleration against the delayed speed difference.
    std::vector<double> dv, a;
    for (std::size_t i = 0; i < r.obs.y.shape()[0]; i += 7) dv.push_back(r.obs.x(i, 0)), a.push_back(r.obs.y(i, 0));
    double lo = 0, hi = 0;
    for (double x : dv) lo = std::min(lo, x), hi = std::max(hi, x);
    std::vector<double> gx, g_true, g_fit, g_nn;
    const double v_mid = 65, gap_mid = 70;
    std::vector<Stimulus> stim;
    for (int k = 0; k <= 60; ++k) {
        const double x = lo + (hi - lo) * k / 60;
        gx.push_back(x);
        g_true.push_back(r.C / r.delta * x);
        g_fit.push_back(r.learned.lambda_m * x);
        stim.push_back({x, v_mid, gap_mid});
    }
    g_nn.resize(stim.size());
    neural_follower(r.learned)(stim, g_nn);

    if (ImPlot::BeginPlot("Training samples: acceleration vs. delayed speed difference", ImVec2(-1, h))) {
        ImPlot::SetupAxes("leader speed - own speed, Delta ago (ft/s)", "acceleration (ft/s^2)");
        const auto [alo, ahi] = limits_of(a);
        ImPlot::SetupAxesLimits(lo, hi, alo, ahi, ImPlotCond_Always);
        ImPlot::PlotScatter("samples (1 in 7)", dv.data(), a.data(), int(dv.size()),
                            ImPlotSpec(ImPlotProp_Marker, ImPlotMarker_Circle, ImPlotProp_MarkerSize, 1.5f,
                                       ImPlotProp_FillAlpha, 0.35f));
        ImPlot::PlotLine("true: (C/Delta) dv", gx.data(), g_true.data(), int(gx.size()),
                         ImPlotSpec(ImPlotProp_LineWeight, 2.0f));
        ImPlot::PlotLine("learned 1959 law", gx.data(), g_fit.data(), int(gx.size()),
                         ImPlotSpec(ImPlotProp_LineWeight, 2.0f));
        ImPlot::PlotLine("neural network (v = 65 ft/s, gap = 70 ft)", gx.data(), g_nn.data(), int(gx.size()),
                         ImPlotSpec(ImPlotProp_LineWeight, 2.0f));
        ImPlot::EndPlot();
    }
}

// ---- Freeway (CTM) -----------------------------------------------------------------------------

struct FreewayPage {
    int seed = 7;
    int day_index = 1;

    // The day (of the first `n`) with the longest travel time.
    void pick_worst_day(int n) {
        const auto all = traffic::simulate_days(corridor, std::size_t(n), unsigned(seed));
        float worst = 0;
        for (int i = 0; i < n; ++i)
            if (float m = std::ranges::max(all[std::size_t(i)].travel_time); m > worst) worst = m, day_index = i;
        dirty = true;
    }
    traffic::Corridor corridor;
    std::vector<traffic::Day> days;
    std::vector<float> heat;   // (detectors, 288), downstream detector first
    bool dirty = true;

    void draw() {
        if (dirty) {
            days = traffic::simulate_days(corridor, std::size_t(day_index + 1), unsigned(seed));
            const auto& d = days.back();
            const std::size_t D = corridor.detectors();
            heat.assign(D * traffic::Day::intervals, 0);
            for (std::size_t k = 0; k < D; ++k)
                for (std::size_t t = 0; t < traffic::Day::intervals; ++t)
                    heat[(D - 1 - k) * traffic::Day::intervals + t] = d.speed[t][k];
            dirty = false;
        }
        const auto& d = days.back();
        ImGui::SetNextItemWidth(220);
        dirty |= ImGui::SliderInt("seed", &seed, 1, 100);
        ImGui::SameLine(0, 30);
        ImGui::SetNextItemWidth(220);
        dirty |= ImGui::SliderInt("day", &day_index, 0, 13);
        ImGui::SameLine(0, 30);
        ImGui::Text("%s%s; longest travel time %.1f min", d.weekend ? "weekend" : "weekday",
                    d.incident ? ", with an incident" : "", std::ranges::max(d.travel_time));

        const float h = ImGui::GetContentRegionAvail().y - 8;
        const std::size_t D = corridor.detectors();
        ImPlot::PushColormap(ImPlotColormap_RdBu);
        if (ImPlot::BeginPlot("Speed (km/h) on a 15 km corridor with a lane drop at km 12-13", ImVec2(-90, h * 0.62f),
                              ImPlotFlags_NoLegend)) {
            ImPlot::SetupAxes("time of day (h)", "position (km)");
            ImPlot::SetupAxesLimits(0, 24, 0, double(D), ImPlotCond_Always);
            ImPlot::PlotHeatmap("speed", heat.data(), int(D), int(traffic::Day::intervals), 0, 110, nullptr,
                                ImPlotPoint(0, 0), ImPlotPoint(24, double(D)));
            ImPlot::EndPlot();
        }
        ImGui::SameLine();
        ImPlot::ColormapScale("km/h", 0, 110, ImVec2(80, h * 0.62f));
        ImPlot::PopColormap();

        std::vector<double> t, tt;
        for (std::size_t i = 0; i < traffic::Day::intervals; ++i) t.push_back(i / 12.0), tt.push_back(d.travel_time[i]);
        if (ImPlot::BeginPlot("Experienced travel time over the corridor", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("departure time (h)", "minutes", 0, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, 24, ImPlotCond_Always);
            ImPlot::PlotLine("travel time", t.data(), tt.data(), int(t.size()));
            ImPlot::EndPlot();
        }
    }
};

// ---- the application ---------------------------------------------------------------------------

struct App {
    CarFollowingPage car_following;
    FreewayPage freeway;
    int select_tab = -1;   // for screenshots: force a tab

    void draw() {
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("XiNN Lab", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBringToFrontOnFocus);
        if (ImGui::BeginTabBar("tabs")) {
            auto tab = [&](const char* name, int index) {
                return ImGui::BeginTabItem(name, nullptr, select_tab == index ? ImGuiTabItemFlags_SetSelected : 0);
            };
            if (tab("Car following", 0)) { car_following.draw(); ImGui::EndTabItem(); }
            if (tab("Training data", 1)) { draw_training_data(car_following); ImGui::EndTabItem(); }
            if (tab("Freeway (CTM)", 2)) { freeway.draw(); ImGui::EndTabItem(); }
            ImGui::EndTabBar();
        }
        ImGui::End();
    }
};

bool save_screenshot(const std::filesystem::path& path, int w, int h) {
    std::vector<std::uint8_t> px(std::size_t(w) * h * 4), flipped(px.size());
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    for (int y = 0; y < h; ++y)
        std::copy_n(px.begin() + std::ptrdiff_t(y) * w * 4, w * 4, flipped.begin() + std::ptrdiff_t(h - 1 - y) * w * 4);
    for (std::size_t i = 3; i < flipped.size(); i += 4) flipped[i] = 255;   // opaque
    return write_png(path, w, h, flipped);
}

}  // namespace lab

int main(int argc, char** argv) {
    std::filesystem::path shots;
    if (argc > 2 && std::string(argv[1]) == "--screenshots") shots = argv[2];

    if (!glfwInit()) return std::println("glfwInit failed"), 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (!shots.empty()) glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* win = glfwCreateWindow(1400, 860, "XiNN Lab", nullptr, nullptr);
    if (!win) return std::println("cannot open an OpenGL 3.3 window"), 1;
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    for (const char* font : {"C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"})
        if (std::filesystem::exists(font)) { io.Fonts->AddFontFromFileTTF(font, 17.0f); break; }
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    lab::App app;

    // Screenshot mode: a fixed sequence of pages.
    struct Shot { const char* file; int tab; std::function<void(lab::App&)> setup; };
    std::vector<Shot> plan;
    if (!shots.empty()) {
        std::filesystem::create_directories(shots);
        using P = lab::CarFollowingPage;
        plan = {
            {"lab_car_following_C0.40.png", 0, [](lab::App& a) { a.car_following.C = 0.40f; a.car_following.dirty = true; }},
            {"lab_car_following_C0.75.png", 0, [](lab::App& a) { a.car_following.C = 0.75f; a.car_following.dirty = true; }},
            {"lab_training_data.png", 1, [](lab::App& a) { a.car_following.learn_now(false); }},
            {"lab_car_following_neural.png", 0,
             [](lab::App& a) { a.car_following.driver = P::Neural; a.car_following.dirty = true; }},
            {"lab_figure6.png", 0, [](lab::App& a) {
                 auto& c = a.car_following;
                 c.driver = P::True, c.C = 0.8f, c.cars = 9, c.delta = 2.0f, c.u = c.gap = 40, c.duration = 30;
                 c.dirty = true;
             }},
            {"lab_freeway.png", 2, [](lab::App& a) { a.freeway.pick_worst_day(14); }},
        };
    }
    std::size_t shot = 0;
    int frames_on_shot = 0;
    if (!plan.empty()) plan[0].setup(app), app.select_tab = plan[0].tab;

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        app.draw();
        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.08f, 0.08f, 0.1f, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (!plan.empty() && ++frames_on_shot == 6) {   // a few frames, so auto-fit plots settle
            const auto path = shots / plan[shot].file;
            std::println("{}: {}", lab::save_screenshot(path, w, h) ? "saved" : "FAILED", path.string());
            if (++shot == plan.size()) break;
            plan[shot].setup(app);
            app.select_tab = plan[shot].tab;
            frames_on_shot = 0;
        }
        if (frames_on_shot == 2) app.select_tab = -1;
        glfwSwapBuffers(win);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
}
