/* SPDX-License-Identifier: BSD-3-Clause */
/* Just enough OpenCL C, in plain C, to run XiNN's kernels on the CPU.
 *
 * The kernels XiNN generates use a small part of OpenCL C, and that part is
 * also C99: qualifiers, uint, work-item ids, barrier() and a few math
 * functions. With this header in front, a C compiler builds the kernel as an
 * ordinary function, and a launcher calls it once per work item:
 *
 *   xcl_launch_1d(item, global)            one work item after another
 *   xcl_launch_2d(item, gx, gy, lx, ly)    work groups one at a time; the
 *                                          items of a group are threads that
 *                                          meet at barrier()
 *
 * __local becomes `static`: the arrays are then shared by the threads of
 * the group that is running, as local memory is shared by a work group.
 * This checks that the source compiles and computes the right numbers. It
 * says nothing about speed, and it is not the OpenCL compiler of a GPU. */
#pragma once

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef unsigned int uint;

#define __kernel
#define __global
#define __local static
#define CLK_LOCAL_MEM_FENCE 1
#define clamp(x, lo, hi) fminf(fmaxf((x), (lo)), (hi))

/* float4 with .x .y .z .w, and vload4(offset, p): p[4 * offset] onwards. */
typedef struct { float x, y, z, w; } float4;
static float4 vload4(size_t offset, const float* p) {
    const float4 v = {p[4 * offset], p[4 * offset + 1], p[4 * offset + 2], p[4 * offset + 3]};
    return v;
}

static _Thread_local size_t xcl_group[2], xcl_local[2];
static size_t xcl_size[2] = {1, 1};
static pthread_barrier_t xcl_barrier;

static size_t get_local_id(uint d) { return xcl_local[d]; }
static size_t get_group_id(uint d) { return xcl_group[d]; }
static size_t get_global_id(uint d) { return xcl_group[d] * xcl_size[d] + xcl_local[d]; }
static void barrier(int fence) { (void)fence; pthread_barrier_wait(&xcl_barrier); }

static void xcl_launch_1d(void (*item)(void), size_t global) {
    xcl_size[0] = 1;
    for (size_t g = 0; g < global; ++g) {
        xcl_group[0] = g;
        xcl_local[0] = 0;
        item();
    }
}

struct xcl_item { void (*item)(void); size_t group[2], local[2]; };

static void* xcl_thread(void* p) {
    const struct xcl_item* it = (const struct xcl_item*)p;
    xcl_group[0] = it->group[0], xcl_group[1] = it->group[1];
    xcl_local[0] = it->local[0], xcl_local[1] = it->local[1];
    it->item();
    return NULL;
}

static void xcl_launch_2d(void (*item)(void), size_t gx, size_t gy, size_t lx, size_t ly) {
    const size_t n = lx * ly;
    pthread_t* threads = (pthread_t*)malloc(n * sizeof *threads);
    struct xcl_item* items = (struct xcl_item*)malloc(n * sizeof *items);
    xcl_size[0] = lx, xcl_size[1] = ly;
    pthread_barrier_init(&xcl_barrier, NULL, (unsigned)n);
    for (size_t by = 0; by < gy / ly; ++by)
        for (size_t bx = 0; bx < gx / lx; ++bx) {
            for (size_t k = 0; k < n; ++k) {
                struct xcl_item it = {item, {bx, by}, {k % lx, k / lx}};
                items[k] = it;
                pthread_create(&threads[k], NULL, xcl_thread, &items[k]);
            }
            for (size_t k = 0; k < n; ++k) pthread_join(threads[k], NULL);
        }
    pthread_barrier_destroy(&xcl_barrier);
    free(items);
    free(threads);
}

/* 0 if every element is within tol (relative to 1 + |want|), else 1. */
static int xcl_compare(const float* got, const float* want, size_t n, float tol) {
    for (size_t i = 0; i < n; ++i)
        if (!(fabsf(got[i] - want[i]) <= tol * (1.0f + fabsf(want[i])))) {
            fprintf(stderr, "element %zu: got %g, want %g\n", i, (double)got[i], (double)want[i]);
            return 1;
        }
    return 0;
}
