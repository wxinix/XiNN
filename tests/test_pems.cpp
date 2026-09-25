// SPDX-License-Identifier: BSD-3-Clause
// The PeMS reference solution, tested on small files written in PeMS formats.
#include "check.hpp"

#include <filesystem>
#include <fstream>
#include <solutions/pems.hpp>

namespace {

std::filesystem::path temp(std::string_view name) { return std::filesystem::temp_directory_path() / name; }

// Three southbound mainline stations on I-5, one ramp, one other freeway.
void write_metadata(const std::filesystem::path& p) {
    std::ofstream f(p);
    f << "ID\tFwy\tDir\tDistrict\tCounty\tCity\tState_PM\tAbs_PM\tLatitude\tLongitude\tLength\tType\tLanes\tName\n"
      << "1001\t5\tS\t7\t37\t\t10.0\t10.0\t34.0\t-118.0\t0.5\tML\t4\tA St\n"
      << "1002\t5\tS\t7\t37\t\t11.0\t11.0\t34.0\t-118.0\t0.5\tML\t4\tB St\n"
      << "1003\t5\tS\t7\t37\t\t12.5\t12.5\t34.0\t-118.0\t0.5\tML\t3\tC St\n"
      << "1004\t5\tS\t7\t37\t\t11.2\t11.2\t34.0\t-118.0\t\tOR\t1\tB St on-ramp\n"
      << "2001\t405\tN\t7\t37\t\t3.0\t3.0\t34.0\t-118.0\t0.5\tML\t5\tOther\n";
}

// One day of 5-minute rows: speed `mph` everywhere, except that station
// 1002 has no data at 00:10 and 1003 is at 30 mph from 01:00.
void write_day(const std::filesystem::path& p, std::string_view date, double mph) {
    std::ofstream f(p);
    for (int i = 0; i < 288; ++i)
        for (long id : {1001L, 1002L, 1003L, 2001L}) {
            if (id == 1002 && i == 2) continue;
            const double v = (id == 1003 && i >= 12) ? 30 : mph;
            f << std::format("{} {:02}:{:02}:00,{},7,5,S,ML,.5,40,100,120,.05,{},10,100,.05,{},1\n", date, i / 12,
                             (i % 12) * 5, id, v, v);
        }
}

}  // namespace

namespace tests {

void metadata_columns_are_found_by_name() {
    write_metadata(temp("xinn_meta.txt"));
    auto m = pems::read_metadata(temp("xinn_meta.txt"));
    check::that(m.has_value());
    check::equal(m->size(), 5uz);
    check::equal((*m)[2].abs_pm, 12.5);
    check::equal((*m)[3].type, std::string("OR"));
}

void southbound_corridor_runs_towards_smaller_postmiles() {
    write_metadata(temp("xinn_meta.txt"));
    auto m = pems::read_metadata(temp("xinn_meta.txt"));
    auto c = pems::build_corridor(*m, 5, 'S', 9, 13);
    check::that(c.has_value());
    check::equal(c->stations.size(), 3uz);   // the ramp and the other freeway are left out
    check::equal(c->stations[0].id, 1003L);
    check::equal(c->stations[2].id, 1001L);
    // Sections reach halfway to the neighbours: 0.75, 1.25, 0.5 miles.
    check::near(c->section_km[0], 0.75 * 1.609344);
    check::near(c->section_km[1], 1.25 * 1.609344);
    check::near(c->length_km(), 2.5 * 1.609344);
}

void five_minute_rows_become_a_tensor() {
    write_metadata(temp("xinn_meta.txt"));
    write_day(temp("xinn_d1.txt"), "03/04/2024", 60);
    write_day(temp("xinn_d2.txt"), "03/05/2024", 60);
    auto m = pems::read_metadata(temp("xinn_meta.txt"));
    auto c = pems::build_corridor(*m, 5, 'S', 9, 13);
    auto s = pems::read_station_5min(*c, {temp("xinn_d1.txt"), temp("xinn_d2.txt")});
    check::that(s.has_value());
    check::equal(s->days, 2uz);
    check::equal(s->data.shape(), xinn::Shape(576, 3, 3));
    check::near(s->data(0, 2, 0), 120 * 12);           // station 1001 is last in corridor order; veh/h
    check::near(s->data(0, 2, 2), 60 * 1.609344, 1e-4);   // km/h
    check::that(std::isnan(s->data(2, 1, 2)));          // 1002 at 00:10: no row

    const std::size_t left = pems::fill_gaps(*s);
    check::equal(left, 0uz);
    check::near(s->data(2, 1, 2), 60 * 1.609344, 1e-4);   // carried forward
}

void trajectory_method_travel_time() {
    write_metadata(temp("xinn_meta.txt"));
    write_day(temp("xinn_d1.txt"), "03/04/2024", 60);
    auto m = pems::read_metadata(temp("xinn_meta.txt"));
    auto c = pems::build_corridor(*m, 5, 'S', 9, 13);
    auto s = pems::read_station_5min(*c, {temp("xinn_d1.txt")});
    pems::fill_gaps(*s);

    // Before 01:00: 2.5 miles at 60 mph = 2.5 minutes.
    check::near(pems::experienced_travel_time(*s, *c, 0), 2.5, 1e-4);
    // From 01:00 the first section (0.75 mi) is at 30 mph: 1.5 min, then 1.75 mi at 60 mph.
    check::near(pems::experienced_travel_time(*s, *c, 12), 1.5 + 1.75, 1e-4);
    // A departure at 00:57 would start at 60 mph and slow down at 01:00 -- handled
    // mid-section. Interval 11 starts at 00:55: 2.5 min at 60 mph ends at 00:57:30,
    // but the first section alone (0.75 mi) is done at 00:55:45, before 01:00.
    check::near(pems::experienced_travel_time(*s, *c, 11), 2.5, 1e-4);
    // Past the end of the data: no answer.
    check::that(std::isnan(pems::experienced_travel_time(*s, *c, 288)));
}

void boundary_crossing_mid_section() {
    // A single long section: 10 km, 60 km/h for the first interval, then 30 km/h.
    pems::Corridor c;
    c.stations.resize(1);
    c.section_km = {10.0};
    pems::Series s;
    s.days = 1;
    s.data = xinn::Tensor<float, 3>(xinn::Shape(288, 1, 3), 30.0f);
    s.data.set(0, 0, 2, 60.0f);
    // 5 min at 60 km/h covers 5 km; the other 5 km at 30 km/h take 10 min.
    check::near(pems::experienced_travel_time(s, c, 0), 15.0, 1e-4);
}

}  // namespace tests

int main() { return check::run_tests<^^tests>(); }
