// SPDX-License-Identifier: BSD-3-Clause
// Reference solution for the chapter 5 project: a real corridor from
// Caltrans PeMS (https://pems.dot.ca.gov, free registration).
//
// Two kinds of PeMS file, both as downloaded from the Data Clearinghouse
// (decompress the .gz files first, e.g. `gzip -d *.gz`):
//
//   Station Metadata   tab-separated, with a header row:
//                      ID  Fwy  Dir  District  County  City  State_PM  Abs_PM
//                      Latitude  Longitude  Length  Type  Lanes  Name  ...
//                      Columns are found by name, so versions that add or
//                      reorder columns still load.
//
//   Station 5-Minute   comma-separated, no header, one row per station and
//                      interval:
//                      Timestamp (MM/DD/YYYY HH:MM:SS), Station, District,
//                      Freeway, Direction, Lane Type, Station Length,
//                      Samples, % Observed, Total Flow, Avg Occupancy,
//                      Avg Speed, then per-lane fields (ignored here).
//                      Flow is veh/5 min over all lanes, occupancy a
//                      fraction, speed mph; any field may be empty.
//
// The column layout follows the PeMS documentation; check it against your
// own files (see the project text in chapter 5).
#pragma once

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

#include <xinn/tensor.hpp>

namespace pems {

inline constexpr float missing = std::numeric_limits<float>::quiet_NaN();

// ---- parsing helpers ------------------------------------------------------------------------

inline std::vector<std::string_view> split(std::string_view line, char sep) {
    std::vector<std::string_view> out;
    for (auto part : line | std::views::split(sep)) out.emplace_back(part.begin(), part.end());
    return out;
}

inline std::optional<double> to_number(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\r')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\r')) s.remove_suffix(1);
    if (s.empty()) return std::nullopt;
    double v{};
    auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || end != s.data() + s.size()) return std::nullopt;
    return v;
}

// "MM/DD/YYYY HH:MM:SS" -> (day number since 1970-01-01, minute of the day).
inline std::optional<std::pair<long, int>> parse_timestamp(std::string_view s) {
    int mo, d, y, h, mi, sec;
    if (s.size() < 19 || std::sscanf(std::string(s).c_str(), "%d/%d/%d %d:%d:%d", &mo, &d, &y, &h, &mi, &sec) != 6)
        return std::nullopt;
    const std::chrono::year_month_day date{std::chrono::year{y}, std::chrono::month{unsigned(mo)},
                                           std::chrono::day{unsigned(d)}};
    if (!date.ok()) return std::nullopt;
    return std::pair{long(std::chrono::sys_days{date}.time_since_epoch().count()), h * 60 + mi};
}

// ---- stations ---------------------------------------------------------------------------------

struct Station {
    long id = 0;
    int freeway = 0;
    char direction = '?';   // N, S, E, W
    double abs_pm = 0;      // absolute postmile, miles
    std::string type;       // ML = mainline, OR/FR = on/off ramp, HV = HOV, ...
    int lanes = 0;
    std::string name;
};

inline std::expected<std::vector<Station>, std::string> read_metadata(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) return std::unexpected(std::format("cannot open {}", path.string()));
    std::string line;
    if (!std::getline(in, line)) return std::unexpected("empty metadata file");
    std::map<std::string, std::size_t, std::less<>> col;
    const auto header = split(line, '\t');
    for (std::size_t i = 0; i < header.size(); ++i) {
        std::string h(header[i]);
        std::erase(h, '\r');
        col[h] = i;
    }
    for (std::string_view need : {"ID", "Fwy", "Dir", "Abs_PM", "Type"})
        if (!col.contains(need)) return std::unexpected(std::format("metadata has no '{}' column", need));

    std::vector<Station> out;
    while (std::getline(in, line)) {
        const auto f = split(line, '\t');
        auto get = [&](std::string_view name) -> std::string_view {
            auto it = col.find(name);
            return it != col.end() && it->second < f.size() ? f[it->second] : std::string_view{};
        };
        const auto id = to_number(get("ID")), fwy = to_number(get("Fwy")), pm = to_number(get("Abs_PM"));
        if (!id || !fwy || !pm || get("Dir").empty()) continue;
        Station s;
        s.id = long(*id);
        s.freeway = int(*fwy);
        s.direction = get("Dir").front();
        s.abs_pm = *pm;
        s.type = std::string(get("Type"));
        std::erase(s.type, '\r');
        s.lanes = int(to_number(get("Lanes")).value_or(0));
        s.name = std::string(get("Name"));
        out.push_back(std::move(s));
    }
    return out;
}

// ---- the corridor -----------------------------------------------------------------------------

struct Corridor {
    std::vector<Station> stations;         // in the direction of travel
    std::vector<double> section_km;        // the stretch each station stands for
    double length_km() const { return std::ranges::fold_left(section_km, 0.0, std::plus{}); }
};

// Mainline stations of one freeway and direction between two postmiles, in
// the order a vehicle meets them. Postmiles grow northbound and eastbound,
// so southbound and westbound corridors run towards smaller postmiles. Each
// station stands for the road halfway to its neighbours; the corridor runs
// from the first station to the last.
inline std::expected<Corridor, std::string> build_corridor(const std::vector<Station>& all, int freeway,
                                                           char direction, double pm_from, double pm_to) {
    Corridor c;
    const double lo = std::min(pm_from, pm_to), hi = std::max(pm_from, pm_to);
    for (const auto& s : all)
        if (s.freeway == freeway && s.direction == direction && s.type == "ML" && s.abs_pm >= lo && s.abs_pm <= hi)
            c.stations.push_back(s);
    if (c.stations.size() < 2)
        return std::unexpected(std::format("fewer than 2 mainline stations on {}-{} between PM {} and {}", freeway,
                                           direction, lo, hi));
    const bool increasing = direction == 'N' || direction == 'E';
    std::ranges::sort(c.stations, [&](const Station& a, const Station& b) {
        return increasing ? a.abs_pm < b.abs_pm : a.abs_pm > b.abs_pm;
    });
    constexpr double km_per_mile = 1.609344;
    const std::size_t n = c.stations.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double before = i > 0 ? std::abs(c.stations[i].abs_pm - c.stations[i - 1].abs_pm) / 2 : 0.0;
        const double after = i + 1 < n ? std::abs(c.stations[i + 1].abs_pm - c.stations[i].abs_pm) / 2 : 0.0;
        c.section_km.push_back((before + after) * km_per_mile);
    }
    return c;
}

// ---- 5-minute data ------------------------------------------------------------------------------

struct Series {
    long first_day = 0;                 // days since 1970-01-01
    std::size_t days = 0;
    // (interval, station, [flow veh/h, occupancy, speed km/h]), in corridor order
    xinn::Tensor<float, 3> data;
    // (interval, station): PeMS's "% Observed", 0-100; missing where no row was found
    xinn::Matrix<float> observed;
};

// Read station 5-minute files for the corridor's stations. Rows of other
// stations are skipped, so whole-district files can be passed as they are.
inline std::expected<Series, std::string> read_station_5min(const Corridor& c,
                                                            const std::vector<std::filesystem::path>& files) {
    std::map<long, std::size_t> index;
    for (std::size_t i = 0; i < c.stations.size(); ++i) index[c.stations[i].id] = i;

    struct Row { long day; int minute; std::size_t station; float flow, occ, speed, observed; };
    std::vector<Row> rows;
    long first = std::numeric_limits<long>::max(), last = std::numeric_limits<long>::min();
    for (const auto& path : files) {
        std::ifstream in(path);
        if (!in) return std::unexpected(std::format("cannot open {}", path.string()));
        std::string line;
        while (std::getline(in, line)) {
            const auto f = split(line, ',');
            if (f.size() < 12) continue;
            const auto id = to_number(f[1]);
            if (!id) continue;
            auto it = index.find(long(*id));
            if (it == index.end()) continue;
            const auto ts = parse_timestamp(f[0]);
            if (!ts) return std::unexpected(std::format("{}: bad timestamp '{}'", path.string(), f[0]));
            auto num = [&](std::size_t i) { return float(to_number(f[i]).value_or(std::nan(""))); };
            rows.push_back({ts->first, ts->second, it->second, num(9), num(10), num(11), num(8)});
            first = std::min(first, ts->first);
            last = std::max(last, ts->first);
        }
    }
    if (rows.empty()) return std::unexpected("no rows for the corridor's stations");

    Series s;
    s.first_day = first;
    s.days = std::size_t(last - first + 1);
    const std::size_t T = s.days * 288, D = c.stations.size();
    s.data = xinn::Tensor<float, 3>(xinn::Shape{T, D, std::size_t{3}}, missing);
    s.observed = xinn::Matrix<float>(xinn::Shape{T, D}, missing);
    auto v = s.data.mut();
    auto o = s.observed.mut();
    for (const auto& r : rows) {
        const std::size_t t = std::size_t(r.day - first) * 288 + std::size_t(r.minute / 5);
        v[t, r.station, 0] = r.flow * 12;               // veh/5 min -> veh/h
        v[t, r.station, 1] = r.occ;
        v[t, r.station, 2] = r.speed * 1.609344f;       // mph -> km/h
        o[t, r.station] = r.observed;
    }
    return s;
}

// Fill short gaps (up to max_gap intervals) by carrying the last value
// forward. Returns the number of values still missing.
inline std::size_t fill_gaps(Series& s, std::size_t max_gap = 3) {
    auto v = s.data.mut();
    std::size_t left = 0;
    for (std::size_t d = 0; d < v.extent(1); ++d)
        for (std::size_t f = 0; f < 3; ++f) {
            std::size_t gap = 0;
            for (std::size_t t = 0; t < v.extent(0); ++t) {
                if (!std::isnan(v[t, d, f])) { gap = 0; continue; }
                if (t > 0 && ++gap <= max_gap && !std::isnan(v[t - 1, d, f])) v[t, d, f] = v[t - 1, d, f];
                else ++left;
            }
        }
    return left;
}

// ---- travel times: the trajectory method -----------------------------------------------------------

// Experienced travel time (minutes) of a vehicle entering at the start of
// interval t0: it drives through each section at that section's speed during
// the interval it is in, switching speed when an interval ends mid-section.
// Returns NaN if a needed speed is missing or the trip leaves the data.
inline float experienced_travel_time(const Series& s, const Corridor& c, std::size_t t0) {
    const auto v = s.data.view();
    const std::size_t T = v.extent(0);
    double clock = double(t0) * 5.0;   // minutes since the first interval
    for (std::size_t i = 0; i < c.section_km.size(); ++i) {
        double left = c.section_km[i];
        while (left > 1e-9) {
            const std::size_t t = std::size_t(clock / 5.0);
            if (t >= T) return missing;
            const float speed = v[t, i, 2];
            if (std::isnan(speed) || speed < 1) return missing;
            const double to_boundary = (double(t + 1) * 5.0 - clock) * speed / 60.0;   // km until the interval ends
            if (to_boundary >= left) { clock += left / speed * 60.0; left = 0; }
            else { clock = double(t + 1) * 5.0; left -= to_boundary; }
        }
    }
    return float(clock - double(t0) * 5.0);
}

}  // namespace pems
