#include "stepper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <omp.h>

//ldoc on
/**
 * ## Implementation
 *
 * ### Structure allocation
 */

central2d_t* central2d_init(float w, float h, int nx, int ny,
                            int nfield, flux_t flux, speed_t speed,
                            float cfl)
{
    // We extend to a four cell buffer to avoid BC comm on odd time steps
    int ng = 4;

    central2d_t* sim = (central2d_t*) malloc(sizeof(central2d_t));
    sim->nx = nx;
    sim->ny = ny;
    sim->ng = ng;
    sim->nfield = nfield;
    sim->dx = w/nx;
    sim->dy = h/ny;
    sim->flux = flux;
    sim->speed = speed;
    sim->cfl = cfl;

    int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;
    int nc = nx_all * ny_all;
    int N  = nfield * nc;
    sim->u  = (float*) malloc((4*N + 6*nx_all)* sizeof(float));
    sim->v  = sim->u +   N;
    sim->f  = sim->u + 2*N;
    sim->g  = sim->u + 3*N;
    sim->scratch = sim->u + 4*N;

    return sim;
}


void central2d_free(central2d_t* sim)
{
    free(sim->u);
    free(sim);
}


int central2d_offset(central2d_t* sim, int k, int ix, int iy)
{
    int nx = sim->nx, ny = sim->ny, ng = sim->ng;
    int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;
    return (k*ny_all+(ng+iy))*nx_all+(ng+ix);
}


/**
 * ### Boundary conditions
 *
 * In finite volume methods, boundary conditions are typically applied by
 * setting appropriate values in ghost cells.  For our framework, we will
 * apply periodic boundary conditions; that is, waves that exit one side
 * of the domain will enter from the other side.
 *
 * We apply the conditions by assuming that the cells with coordinates
 * `nghost <= ix <= nx+nghost` and `nghost <= iy <= ny+nghost` are
 * "canonical", and setting the values for all other cells `(ix,iy)`
 * to the corresponding canonical values `(ix+p*nx,iy+q*ny)` for some
 * integers `p` and `q`.
 */

static inline
void copy_subgrid(float* restrict dst,
                  const float* restrict src,
                  int nx, int ny, int stride)
{
    for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix)
            dst[iy*stride+ix] = src[iy*stride+ix];
}

void central2d_periodic(float* restrict u,
                        int nx, int ny, int ng, int nfield)
{
    // Stride and number per field
    int s = nx + 2*ng;
    int field_stride = (ny+2*ng)*s;

    // Offsets of left, right, top, and bottom data blocks and ghost blocks
    int l = nx,   lg = 0;
    int r = ng,   rg = nx+ng;
    int b = ny*s, bg = 0;
    int t = ng*s, tg = (ny+ng)*s;

    // Copy data into ghost cells on each side
    for (int k = 0; k < nfield; ++k) {
        float* uk = u + k*field_stride;
        copy_subgrid(uk+lg, uk+l, ng, ny+2*ng, s);
        copy_subgrid(uk+rg, uk+r, ng, ny+2*ng, s);
        copy_subgrid(uk+tg, uk+t, nx+2*ng, ng, s);
        copy_subgrid(uk+bg, uk+b, nx+2*ng, ng, s);
    }
}

/**
 * Do left and right periodic BC only
 * This is for updating x BC for subdomains
 */

void central2d_periodic_x(float *restrict u,
                        int nx, int ny, int ng, int nfield,
                        int field_stride)
{
    // Stride and number per field
    int s = nx + 2 * ng;

    // Offsets of left, right, top, and bottom data blocks and ghost blocks
    int l = nx, lg = 0;
    int r = ng, rg = nx + ng;

    // Copy data into ghost cells on each side
    for (int k = 0; k < nfield; ++k)
    {
        float *uk = u + k * field_stride;
        copy_subgrid(uk + lg, uk + l, ng, ny, s);
        copy_subgrid(uk + rg, uk + r, ng, ny, s);
    }
}

/**
 * ### Derivatives with limiters
 *
 * In order to advance the time step, we also need to estimate
 * derivatives of the fluxes and the solution values at each cell.
 * In order to maintain stability, we apply a limiter here.
 *
 * The minmod limiter *looks* like it should be expensive to computer,
 * since superficially it seems to require a number of branches.
 * We do something a little tricky, getting rid of the condition
 * on the sign of the arguments using the `copysign` instruction.
 * If the compiler does the "right" thing with `max` and `min`
 * for floating point arguments (translating them to branch-free
 * intrinsic operations), this implementation should be relatively fast.
 */


// Branch-free computation of minmod of two numbers times 2s
static inline
float xmin2s(float s, float a, float b) {
    float sa = copysignf(s, a);
    float sb = copysignf(s, b);
    float abs_a = fabsf(a);
    float abs_b = fabsf(b);
    float min_abs = (abs_a < abs_b ? abs_a : abs_b);
    return (sa+sb) * min_abs;
}


// Limited combined slope estimate
static inline
float limdiff(float um, float u0, float up) {
    const float theta = 2.0;
    const float quarter = 0.25;
    float du1 = u0-um;   // Difference to left
    float du2 = up-u0;   // Difference to right
    float duc = up-um;   // Twice centered difference
    return xmin2s( quarter, xmin2s(theta, du1, du2), duc );
}


// Compute limited derivs
static inline
void limited_deriv1(float* restrict du,
                    const float* restrict u,
                    int ncell)
{
    // #pragma omp parallel for
    for (int i = 0; i < ncell; ++i)
        du[i] = limdiff(u[i-1], u[i], u[i+1]);
}


// Compute limited derivs across stride
static inline
void limited_derivk(float* restrict du,
                    const float* restrict u,
                    int ncell, int stride)
{
    assert(stride > 0);
    // #pragma omp parallel for
    for (int i = 0; i < ncell; ++i)
        du[i] = limdiff(u[i-stride], u[i], u[i+stride]);
}


/**
 * ### Advancing a time step
 *
 * Take one step of the numerical scheme.  This consists of two pieces:
 * a first-order corrector computed at a half time step, which is used
 * to obtain new $F$ and $G$ values; and a corrector step that computes
 * the solution at the full step.  For full details, we refer to the
 * [Jiang and Tadmor paper][jt].
 *
 * The `compute_step` function takes two arguments: the `io` flag
 * which is the time step modulo 2 (0 if even, 1 if odd); and the `dt`
 * flag, which actually determines the time step length.  We need
 * to know the even-vs-odd distinction because the Jiang-Tadmor
 * scheme alternates between a primary grid (on even steps) and a
 * staggered grid (on odd steps).  This means that the data at $(i,j)$
 * in an even step and the data at $(i,j)$ in an odd step represent
 * values at different locations in space, offset by half a space step
 * in each direction.  Every other step, we shift things back by one
 * mesh cell in each direction, essentially resetting to the primary
 * indexing scheme.
 *
 * We're slightly tricky in the corrector in that we write
 * $$
 *   v(i,j) = (s(i+1,j) + s(i,j)) - (d(i+1,j)-d(i,j))
 * $$
 * where $s(i,j)$ comprises the $u$ and $x$-derivative terms in the
 * update formula, and $d(i,j)$ the $y$-derivative terms.  This cuts
 * the arithmetic cost a little (not that it's that big to start).
 * It also makes it more obvious that we only need four rows worth
 * of scratch space.
 */


// Predictor half-step
static
void central2d_predict(float* restrict v,
                       float* restrict scratch,
                       const float* restrict u,
                       const float* restrict f,
                       const float* restrict g,
                       float dtcdx2, float dtcdy2,
                       int nx, int ny, int nfield)
{
    float* restrict fx = scratch;
    float* restrict gy = scratch+nx;
    // #pragma omp parallel for collapse(2)
    for (int k = 0; k < nfield; ++k) {
        for (int iy = 1; iy < ny-1; ++iy) {
            int offset = (k*ny+iy)*nx+1;
            limited_deriv1(fx+1, f+offset, nx-2);
            limited_derivk(gy+1, g+offset, nx-2, nx);
            for (int ix = 1; ix < nx-1; ++ix) {
                int offset = (k*ny+iy)*nx+ix;
                v[offset] = u[offset] - dtcdx2 * fx[ix] - dtcdy2 * gy[ix];
            }
        }
    }
}


// Corrector
static
void central2d_correct_sd(float* restrict s,
                          float* restrict d,
                          const float* restrict ux,
                          const float* restrict uy,
                          const float* restrict u,
                          const float* restrict f,
                          const float* restrict g,
                          float dtcdx2, float dtcdy2,
                          int xlo, int xhi)
{
    // #pragma omp parallel for
    for (int ix = xlo; ix < xhi; ++ix)
        s[ix] =
            0.2500f * (u [ix] + u [ix+1]) +
            0.0625f * (ux[ix] - ux[ix+1]) +
            dtcdx2  * (f [ix] - f [ix+1]);
    // #pragma omp parallel for
    for (int ix = xlo; ix < xhi; ++ix)
        d[ix] =
            0.0625f * (uy[ix] + uy[ix+1]) +
            dtcdy2  * (g [ix] + g [ix+1]);
}


// Corrector
static
void central2d_correct(float* restrict v,
                       float* restrict scratch,
                       const float* restrict u,
                       const float* restrict f,
                       const float* restrict g,
                       float dtcdx2, float dtcdy2,
                       int xlo, int xhi, int ylo, int yhi,
                       int nx, int ny, int nfield)
{
    assert(0 <= xlo && xlo < xhi && xhi <= nx);
    assert(0 <= ylo && ylo < yhi && yhi <= ny);

    float* restrict ux = scratch;
    float* restrict uy = scratch +   nx;
    float* restrict s0 = scratch + 2*nx;
    float* restrict d0 = scratch + 3*nx;
    float* restrict s1 = scratch + 4*nx;
    float* restrict d1 = scratch + 5*nx;

    for (int k = 0; k < nfield; ++k) {

        float*       restrict vk = v + k*ny*nx;
        const float* restrict uk = u + k*ny*nx;
        const float* restrict fk = f + k*ny*nx;
        const float* restrict gk = g + k*ny*nx;

        limited_deriv1(ux+1, uk+ylo*nx+1, nx-2);
        limited_derivk(uy+1, uk+ylo*nx+1, nx-2, nx);
        central2d_correct_sd(s1, d1, ux, uy,
                             uk + ylo*nx, fk + ylo*nx, gk + ylo*nx,
                             dtcdx2, dtcdy2, xlo, xhi);
        for (int iy = ylo; iy < yhi; ++iy) {

            float* tmp;
            tmp = s0; s0 = s1; s1 = tmp;
            tmp = d0; d0 = d1; d1 = tmp;

            limited_deriv1(ux+1, uk+(iy+1)*nx+1, nx-2);
            limited_derivk(uy+1, uk+(iy+1)*nx+1, nx-2, nx);
            central2d_correct_sd(s1, d1, ux, uy,
                                 uk + (iy+1)*nx, fk + (iy+1)*nx, gk + (iy+1)*nx,
                                 dtcdx2, dtcdy2, xlo, xhi);

            for (int ix = xlo; ix < xhi; ++ix)
                vk[iy*nx+ix] = (s1[ix]+s0[ix])-(d1[ix]-d0[ix]);
        }
    }
}


static
void central2d_step(float* restrict u, float* restrict v,
                    float* restrict scratch,
                    float* restrict f,
                    float* restrict g,
                    int io, int nx, int ny, int ng,
                    int nfield, flux_t flux, speed_t speed,
                    float dt, float dx, float dy)
{
    int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;

    float dtcdx2 = 0.5 * dt / dx;
    float dtcdy2 = 0.5 * dt / dy;

    flux(f, g, u, nx_all * ny_all, nx_all * ny_all);

    central2d_predict(v, scratch, u, f, g, dtcdx2, dtcdy2,
                      nx_all, ny_all, nfield);

    // Flux values of f and g at half step
    for (int iy = 1; iy < ny_all-1; ++iy) {
        int jj = iy*nx_all+1;
        flux(f+jj, g+jj, v+jj, nx_all-2, nx_all * ny_all);
    }

    central2d_correct(v+io*(nx_all+1), scratch, u, f, g, dtcdx2, dtcdy2,
                      ng-io, nx+ng-io,
                      ng-io, ny+ng-io,
                      nx_all, ny_all, nfield);
}


// static void central2d_step_subdomain(float *restrict u, float *restrict v,
//                            float *restrict scratch,
//                            float *restrict f,
//                            float *restrict g,
//                            int io, int nx, int ny, int ng,
//                            int nfield, flux_t flux, speed_t speed,
//                            float dt, float dx, float dy,
//                            int field_stride)
// {
//     int nx_all = nx + 2 * ng;
//     int ny_all = ny + 2 * ng;

//     float dtcdx2 = 0.5 * dt / dx;
//     float dtcdy2 = 0.5 * dt / dy;

//     flux(f, g, u, nx_all * ny_all, nx_all * ny_all);

//     central2d_predict(v, scratch, u, f, g, dtcdx2, dtcdy2,
//                       nx_all, ny_all, nfield);

//     // Flux values of f and g at half step
//     for (int iy = 1; iy < ny_all - 1; ++iy)
//     {
//         int jj = iy * nx_all + 1;
//         flux(f + jj, g + jj, v + jj, nx_all - 2, nx_all * ny_all);
//     }

//     central2d_correct(v + io * (nx_all + 1), scratch, u, f, g, dtcdx2, dtcdy2,
//                       ng - io, nx + ng - io,
//                       ng - io, ny + ng - io,
//                       nx_all, ny_all, nfield);
// }

/**
 * ### Advance a fixed time
 *
 * The `run` method advances from time 0 (initial conditions) to time
 * `tfinal`.  Note that `run` can be called repeatedly; for example,
 * we might want to advance for a period of time, write out a picture,
 * advance more, and write another picture.  In this sense, `tfinal`
 * should be interpreted as an offset from the time represented by
 * the simulator at the start of the call, rather than as an absolute time.
 *
 * We always take an even number of steps so that the solution
 * at the end lives on the main grid instead of the staggered grid.
 */

#ifndef BATCH
#define BATCH 1
#endif

static inline
int* compute_offset(int ny, int np){
    // int ny_inner = ny / np;
    int* offsets = (int*) malloc((np+1) * sizeof(int));
    for (int i = 0; i <= np; ++i){
        long r = i * ny;
        offsets[i] = (int) (r/np);
    }
    return offsets;
}

/**
 * The `sub_copyin` routine updates the ghost paddings into subdomains.
 * The `sub_copyin_meat` routine updates the meats into subdomains.
 * The `sub_copyout` routine moves the range corresponding
 * to `own_start` to `own_end` (starting at offset `own_start-sub_start`)
 * from the local array back into the global array.
 *
 */
void sub_copyin(float *restrict ulocal,
                float *restrict uglobal,
                int own_start,
                int own_end,
                int nfield,
                int nx_all,
                int ny,
                int ng,
                int field_stride_local,
                int field_stride_global)
{
    // deal with the top pad
    if (own_start < BATCH * ng){
        // copy the normal part
        if (own_start > 0){
            for (int j = 0; j < nfield; ++j)
            {
                float *ul = ulocal + field_stride_local * j + (BATCH * ng - own_start) * nx_all;
                float *ug = uglobal + field_stride_global * j;
                memcpy(ul, ug, own_start * nx_all * sizeof(float));
            }
        }
        // copy the pariodic part
        for (int j = 0; j < nfield; ++j)
        {
            float *ul = ulocal + field_stride_local * j;
            float *ug = uglobal + field_stride_global * j + (ny - (BATCH * ng - own_start)) * nx_all;
            memcpy(ul, ug, (BATCH * ng - own_start) * nx_all * sizeof(float));
        }
    } else {
        for (int j = 0; j < nfield; ++j){
            float *ul = ulocal + field_stride_local * j;
            float *ug = uglobal + field_stride_global * j + (own_start - BATCH * ng) * nx_all;
            memcpy(ul, ug, BATCH * ng * nx_all * sizeof(float));
        }
    }
    // deal with the bottom pad
    if (own_end + BATCH * ng > ny) {
        // copy the normal part
        if (ny - own_end > 0) {
            for (int j = 0; j < nfield; ++j) {
                float *ul = ulocal + (own_end - own_start + BATCH * ng) * nx_all + field_stride_local * j;
                float *ug = uglobal + field_stride_global * j + own_end * nx_all;
                memcpy(ul, ug, (ny - own_end) * nx_all * sizeof(float));
            }
        }
        // copy the pariodic part
        for (int j = 0; j < nfield; ++j) {
            float *ul = ulocal + (own_end - own_start + BATCH * ng + ny - own_end) * nx_all + field_stride_local * j;
            float *ug = uglobal + field_stride_global * j;
            memcpy(ul, ug, (BATCH * ng - (ny - own_end)) * nx_all * sizeof(float));
        }
    } else {
        for (int j = 0; j < nfield; ++j){
            float *ul = ulocal + (own_end - own_start + BATCH * ng) * nx_all + field_stride_local * j;
            float *ug = uglobal + field_stride_global * j + own_end * nx_all;
            memcpy(ul, ug, BATCH * ng * nx_all * sizeof(float));
        }
    }
}

void sub_copyout(float *restrict ulocal,
                 float *restrict uglobal,
                 int own_start, int own_end,
                 int nfield,
                 int nx_all,
                 int ny,
                 int ng,
                 int field_stride_local,
                 int field_stride_global) {
    for (int j = 0; j < nfield; ++j){
        float *ul = ulocal + BATCH * ng * nx_all + field_stride_local * j;
        float *ug = uglobal + field_stride_global * j + own_start * nx_all;
        memcpy(ug, ul, (own_end - own_start) * nx_all * sizeof(float));
    }
}

void sub_copyin_meat(float *restrict ulocal,
                     float *restrict uglobal,
                     int own_start, int own_end,
                     int nfield,
                     int nx_all,
                     int ny,
                     int ng,
                     int field_stride_local,
                     int field_stride_global)
{
    for (int j = 0; j < nfield; ++j)
    {
        float *ul = ulocal + BATCH * ng * nx_all + field_stride_local * j;
        float *ug = uglobal + field_stride_global * j + own_start * nx_all;
        memcpy(ul, ug, (own_end - own_start) * nx_all * sizeof(float));
    }
}

static
int central2d_xrun(float* restrict u, float* restrict v,
                   float* restrict scratch,
                   float* restrict f,
                   float* restrict g,
                   int nx, int ny, int ng,
                   int nfield, flux_t flux, speed_t speed,
                   float tfinal, float dx, float dy, float cfl)
{
    int nstep;
    int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;
    bool done = false;
    float t = 0;
    int np = omp_get_max_threads();
    int* offsets = compute_offset(ny, np);
    // printf("BATCH = %d, ", BATCH);
    // printf("np = %d, ", np);
    // printf("offsets = [%d", offsets[0]);
    // for (int i = 1; i <= np; ++i){
    //     printf(", %d", offsets[i]);
    // }
    // printf("]\n");
    int N = nx_all * nfield * (ny + 2 * ng * np * BATCH);
    float* ulocal = (float*) malloc(N * sizeof(float));
    float* vlocal = (float*) malloc(N * sizeof(float));
    float* flocal = (float*) malloc(N * sizeof(float));
    float* glocal = (float*) malloc(N * sizeof(float));
    float* scratch_local = (float*) malloc((6 * nx_all * np) * sizeof(float));
    float *cxs = (float *)malloc(np * sizeof(float));
    float *cys = (float *)malloc(np * sizeof(float));
    int curr_step_in_batch = 0;
    /**
     * David's suggestions:
     * 1. reduce parallel for sections
     * 2. use small cfl, use same dt within a batch
     * 3. use private subdomain struct for each process
     */

    // initialize ulocal
    // central2d_periodic(u, nx, ny, ng, nfield);
#pragma omp parallel for
    for (int i = 0; i < np; ++i)
    {
        sub_copyin(ulocal + nx_all * nfield * (offsets[i] + 2 * i * ng * BATCH), // start from the ith subdomain
                   u + nx_all * ng,                                              // ingore the ghost rows in uglobal
                   offsets[i],
                   offsets[i + 1],
                   nfield,
                   nx_all,
                   ny,
                   ng,
                   nx_all * (2 * ng * BATCH + offsets[i + 1] - offsets[i]),
                   nx_all * ny_all);
        sub_copyin_meat(ulocal + nx_all * nfield * (offsets[i] + 2 * i * ng * BATCH), // start from the ith subdomain
                        u + nx_all * ng,                                              // ingore the ghost rows in uglobal
                        offsets[i],
                        offsets[i + 1],
                        nfield,
                        nx_all,
                        ny,
                        ng,
                        nx_all * (2 * ng * BATCH + offsets[i + 1] - offsets[i]),
                        nx_all * ny_all);
    }

#pragma omp parallel firstprivate(t, done, curr_step_in_batch)
    {
        int i = omp_get_thread_num();
        // printf("thread id = %d\n", i);
        float dt = 1.0e-15f;
        while (!done)
        {
            if (curr_step_in_batch % BATCH == 0)
            {
                // update the global
                central2d_periodic_x(ulocal + nx_all * nfield * (offsets[i] + (2 * i) * ng * BATCH),
                                     nx, offsets[i + 1] - offsets[i] + 2 * ng * BATCH, ng, nfield,
                                     nx_all * (2 * ng * BATCH + offsets[i + 1] - offsets[i]));
                sub_copyout(ulocal + nx_all * nfield * (offsets[i] + 2 * i * ng * BATCH), // start from the ith subdomain
                            u + nx_all * ng,                                              // ingore the ghost rows in uglobal
                            offsets[i],
                            offsets[i + 1],
                            nfield,
                            nx_all,
                            ny,
                            ng,
                            nx_all * (2 * ng * BATCH + offsets[i + 1] - offsets[i]),
                            nx_all * ny_all);
                float cxy[2] = {1.0e-15f, 1.0e-15f};
                speed(
                    cxy,
                    ulocal + nx_all * nfield * (offsets[i] + (2 * i) * ng * BATCH) + nx_all * ng * BATCH,
                    nx_all * (offsets[i + 1] - offsets[i]),
                    nx_all * (offsets[i + 1] - offsets[i] + 2 * ng * BATCH));
                cxs[i] = cxy[0];
                cys[i] = cxy[1];
#pragma omp barrier
                float max_c = 1.0e-15f;
                for (int _i = 0; _i < np; ++_i){
                    if (cxs[_i] / dx > max_c) max_c = cxs[_i] / dx;
                    if (cys[_i] / dx > max_c) max_c = cys[_i] / dx;
                }
                dt = cfl / max_c;
                sub_copyin(ulocal + nx_all * nfield * (offsets[i] + 2 * i * ng * BATCH), // start from the ith subdomain
                            u + nx_all * ng,                                              // ingore the ghost rows in uglobal
                            offsets[i],
                            offsets[i + 1],
                            nfield,
                            nx_all,
                            ny,
                            ng,
                            nx_all * (2 * ng * BATCH + offsets[i + 1] - offsets[i]),
                            nx_all * ny_all);
            }
            central2d_periodic_x(ulocal + nx_all * nfield * (offsets[i] + (2 * i) * ng * BATCH),
                                 nx, offsets[i + 1] - offsets[i] + 2 * ng * BATCH, ng, nfield,
                                 nx_all * (2 * ng * BATCH + offsets[i + 1] - offsets[i]));
            if (t + 2*dt >= tfinal) {
                dt = (tfinal-t)/2;
                done = true;
            }
            int offset_local = nx_all * nfield * (offsets[i] + i * 2 * ng * BATCH);
            int subdomain_ny = offsets[i + 1] - offsets[i] + 2 * ng * (BATCH - 1);
            central2d_step(ulocal + offset_local,
                            vlocal + offset_local,
                            scratch_local + 6 * nx_all * i,
                            flocal + offset_local,
                            glocal + offset_local,
                            0,
                            nx + 4,
                            subdomain_ny + 4,
                            ng - 2, nfield, flux, speed, dt, dx, dy);
            central2d_step(vlocal + offset_local,
                            ulocal + offset_local,
                            scratch_local + 6 * nx_all * i,
                            flocal + offset_local,
                            glocal + offset_local,
                            1,
                            nx,
                            subdomain_ny,
                            ng,
                            nfield, flux, speed, dt, dx, dy);
            // }
            t += 2*dt;
            if (i == 0)
                nstep += 2;
            curr_step_in_batch += 1;
            if (done) {
                sub_copyout(ulocal + nx_all * nfield * (offsets[i] + 2 * i * ng * BATCH), // start from the ith subdomain
                            u + nx_all * ng,                                              // ingore the ghost rows in uglobal
                            offsets[i],
                            offsets[i + 1],
                            nfield,
                            nx_all,
                            ny,
                            ng,
                            nx_all * (2 * ng * BATCH + offsets[i + 1] - offsets[i]),
                            nx_all * ny_all);
#pragma omp barrier
            }
        }
    }
    // clean up and return
    free(ulocal);
    free(vlocal);
    free(flocal);
    free(glocal);
    free(scratch_local);
    free(offsets);
    free(cxs);
    free(cys);
    return nstep;
}


int central2d_run(central2d_t* sim, float tfinal)
{
    return central2d_xrun(sim->u, sim->v, sim->scratch,
                          sim->f, sim->g,
                          sim->nx, sim->ny, sim->ng,
                          sim->nfield, sim->flux, sim->speed,
                          tfinal, sim->dx, sim->dy, sim->cfl);
}