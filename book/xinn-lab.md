# XiNN Lab

XiNN Lab is an interactive front end for the traffic examples. It is built
with **Dear ImGui** (immediate-mode user interfaces) and **ImPlot** (charts
for Dear ImGui), on **GLFW** and OpenGL 3.

![XiNN Lab: a line of 8 cars at C = 0.75](images/lab_car_following_C0.75.png)

## Building and running

The Lab is optional, so the library and its tests need none of this. The
`gui` preset turns it on. CMake then fetches GLFW 3.4, Dear ImGui 1.92.9 and
ImPlot 1.0, each pinned to its release:

```
cmake --preset gui
cmake --build --preset gui --target xinn_lab
build/gui/gui/xinn_lab
```

In CLion, enable the `release + XiNN Lab` profile.

## The three tabs

**Car following.** The model of Herman et al. (1959) from the
[case study](case-car-following.md). Sliders set C, the reaction time Δ, the
number of cars, the initial speed and the spacing. The buttons reproduce the
paper's Figures 3, 5 and 6. The side panel states what theory predicts for the
current C, and what this run shows. Drag C across 1/e, ½ and π/2 and watch
the line of cars change character.

*Learn this driver from data* observes the current driver, trains the two
models of the case study, and makes them selectable as the driver of the
line. The training runs on a `std::jthread`, so the window stays responsive.

**Training data.** Snapshots of what the learned drivers were trained on: an
example episode with the lead car's random manoeuvres and the six cars'
responses, and the cloud of (Δv, acceleration) samples with the true, the
fitted and the neural response drawn over it.

**Freeway (CTM).** A day on the Cell Transmission Model corridor of
chapter 5: the space–time speed map of the 15 detectors, and the travel time
of each departure.

## Screenshots for the book

The figures in this book come from the Lab itself:

```
build/gui/gui/xinn_lab --screenshots book/images
python tools/compress_png.py book/images/*.png
```

`--screenshots` renders each page for a few frames, reads the frame buffer
with `glReadPixels`, and writes it as a PNG. The PNG writer in `gui/png.hpp`
is about 80 lines because it does not compress (zlib "stored" blocks). The
Python script then recompresses the images with the standard library's
`zlib`, from 4.7 MB to about 130 KB each.

## Immediate mode

Dear ImGui redraws the whole interface every frame from the program's own
state. There are no widget objects to create, connect or destroy:

```cpp
dirty |= ImGui::SliderFloat("C", &C, 0.1f, 2.0f, "%.3f");
if (dirty) simulate_now();
if (ImPlot::BeginPlot("Spacing to the car ahead, minus initial spacing")) {
    ImPlot::PlotLine("1-2", t.data(), d.data(), int(t.size()));
    ImPlot::EndPlot();
}
```

A slider returns `true` when it changes its value, so the page knows when to
simulate again. This style suits exploratory tools like this one: the code
for a panel reads top to bottom like the panel itself.
