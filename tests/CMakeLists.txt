# SPDX-License-Identifier: BSD-3-Clause
function(xinn_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE xinn)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# A death test passes when the program terminates (a contract fired).
function(xinn_death_test name)
    xinn_test(${name})
    set_tests_properties(${name} PROPERTIES WILL_FAIL TRUE)
endfunction()

# A compile-fail test passes when compiling the file fails with a message
# matching `regex`.
function(xinn_compile_fail name regex)
    add_test(NAME ${name}
             COMMAND ${CMAKE_CXX_COMPILER} -std=c++26 -freflection -fcontracts -fsyntax-only
                     -I${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/${name}.cpp)
    set_tests_properties(${name} PROPERTIES PASS_REGULAR_EXPRESSION "${regex}")
endfunction()

xinn_test(test_shape)
xinn_test(test_tensor)
xinn_test(test_expr)
xinn_test(test_fusion)
xinn_test(test_autograd)
xinn_test(test_module)
xinn_test(test_io)
xinn_test(test_parallel)
xinn_test(test_conv)
xinn_test(test_attention)
xinn_test(test_train)
target_compile_definitions(test_train PRIVATE XINN_DATA_DIR="${PROJECT_SOURCE_DIR}/data")
xinn_test(test_ctm)
target_include_directories(test_ctm PRIVATE ${PROJECT_SOURCE_DIR}/examples)
xinn_test(test_pems)
xinn_test(test_car_following)
target_include_directories(test_car_following PRIVATE ${PROJECT_SOURCE_DIR}/examples)
target_include_directories(test_pems PRIVATE ${PROJECT_SOURCE_DIR}/examples)

# Check a file written by test_io against the safetensors format, in Python.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    set_tests_properties(test_io PROPERTIES FIXTURES_SETUP safetensors_file)
    add_test(NAME safetensors_spec
             COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/verify_safetensors.py
                     $ENV{TEMP}/xinn_spec_check.safetensors)
    set_tests_properties(safetensors_spec PROPERTIES FIXTURES_REQUIRED safetensors_file)
endif()
xinn_death_test(death_write_shared)
xinn_death_test(death_out_of_bounds)
xinn_death_test(death_broadcast_mismatch)
xinn_death_test(death_matmul_mismatch)
xinn_compile_fail(no_grad_rule "has no gradient rule")
xinn_compile_fail(loss_without_params "depends on no Param")
xinn_test(test_rnn)

# Chapter 7: the GPU backend. Runtime tests skip without an OpenCL device.
xinn_test(test_gpu)
xinn_compile_fail(gpu_op_without_code "has no OpenCL code")
# The generated kernels, compiled as C with a small OpenCL emulation and run
# on the CPU against CPU results: checks the source without a GPU.
add_executable(gpu_emit_c gpu_emulate/emit_c.cpp)
target_link_libraries(gpu_emit_c PRIVATE xinn)
add_test(NAME gpu_kernels_as_c
         COMMAND ${CMAKE_COMMAND} -DEMIT=$<TARGET_FILE:gpu_emit_c> -DCC=${CMAKE_CXX_COMPILER}
                 -DDIR=${CMAKE_CURRENT_BINARY_DIR}/gpu_kernels_as_c
                 -DPRELUDE=${CMAKE_CURRENT_SOURCE_DIR}/gpu_emulate
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/gpu_emulate/run.cmake)
xinn_test(test_gnn)
