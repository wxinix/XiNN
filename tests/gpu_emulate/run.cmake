# SPDX-License-Identifier: BSD-3-Clause
# Compile the kernels written by gpu_emit_c as C, run them, and fail if any
# result differs from the CPU's.
#   cmake -DEMIT=<gpu_emit_c> -DCC=<compiler> -DDIR=<out dir> -DPRELUDE=<dir of prelude.h> -P run.cmake
file(REMOVE_RECURSE ${DIR})
execute_process(COMMAND ${EMIT} ${DIR} RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "gpu_emit_c failed: ${rc}")
endif()

file(GLOB kernels ${DIR}/*.c)
list(LENGTH kernels count)
if(count EQUAL 0)
    message(FATAL_ERROR "no kernels were written")
endif()
foreach(src IN LISTS kernels)
    get_filename_component(name ${src} NAME_WE)
    execute_process(COMMAND ${CC} -x c -std=c11 -O1 -Wall -Wextra -Wno-unused-function
                            -Werror=implicit-function-declaration -I${PRELUDE}
                            ${src} -o ${DIR}/${name}.exe -static -pthread -lm
                    RESULT_VARIABLE rc ERROR_VARIABLE err OUTPUT_VARIABLE out)
    if(NOT rc EQUAL 0 OR NOT err STREQUAL "")
        message(FATAL_ERROR "${name}: the kernel does not compile cleanly as C:\n${out}${err}")
    endif()
    execute_process(COMMAND ${DIR}/${name}.exe RESULT_VARIABLE rc ERROR_VARIABLE err)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "${name}: wrong result: ${err}")
    endif()
    message(STATUS "${name}: compiles, and matches the CPU")
endforeach()
