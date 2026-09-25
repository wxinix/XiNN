# SPDX-License-Identifier: BSD-3-Clause
# XiNN Lab: an interactive front end, built with Dear ImGui and ImPlot on
# GLFW + OpenGL 3. All three are fetched at configure time, pinned to
# releases (MIT, MIT and zlib licenses).
include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.4)
FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG v1.92.9
                     SOURCE_SUBDIR _sources_only)
FetchContent_Declare(implot GIT_REPOSITORY https://github.com/epezent/implot.git GIT_TAG v1.0
                     SOURCE_SUBDIR _sources_only)
FetchContent_MakeAvailable(glfw imgui implot)

add_library(xinn_imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp)
target_include_directories(xinn_imgui SYSTEM PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends ${implot_SOURCE_DIR})
target_link_libraries(xinn_imgui PUBLIC glfw)
if(WIN32)
    target_link_libraries(xinn_imgui PUBLIC opengl32)
else()
    find_package(OpenGL REQUIRED)
    target_link_libraries(xinn_imgui PUBLIC OpenGL::GL)
endif()

add_executable(xinn_lab xinn_lab.cpp)
target_link_libraries(xinn_lab PRIVATE xinn xinn_imgui)
target_include_directories(xinn_lab PRIVATE ${PROJECT_SOURCE_DIR}/examples)
