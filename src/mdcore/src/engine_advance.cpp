/*
 * engine_advance.cpp
 *
 *  Created on: Jan 2, 2021
 *      Author: andy
 */

/* Include conditional headers. */
#include "mdcore_config.h"
#include "engine.h"
#include "engine_advance.h"
#include "errs.h"
#include "MxCluster.hpp"
#include "Flux.hpp"
#include "boundary_eval.hpp"
#include "../../mx_error.h"
#include "../../MxLogger.h"

#include <math.h>

#include <sstream>
#pragma clang diagnostic ignored "-Wwritable-strings"
#include <iostream>

#if MX_THREADING
#include "MxTaskScheduler.hpp"
#endif

#ifdef WITH_MPI
#include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#ifdef WITH_METIS
#include <metis.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

/* the error macro. */
#define error(id) ( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

static int engine_advance_forward_euler ( struct engine *e );
static int engine_advance_runge_kutta_4 ( struct engine *e );



static int _toofast_error(MxParticle *p, int line, const char* func) {
    std::stringstream ss;
    ss << "ERROR, particle moving too fast, p: {" << std::endl;
    ss << "\tid: " << p->id << ", " << std::endl;
    ss << "\ttype: " << _Engine.types[p->typeId].name << "," << std::endl;
    ss << "\tx: [" << p->x[0] << ", " << p->x[1] << ", " << p->x[2] << "], " << std::endl;
    ss << "\tv: [" << p->v[0] << ", " << p->v[1] << ", " << p->v[2] << "], " << std::endl;
    ss << "\tf: [" << p->f[0] << ", " << p->f[1] << ", " << p->f[2] << "], " << std::endl;
    ss << "}";
    
    MxErr_Set(E_FAIL, ss.str().c_str(), line, __FILE__, func);
    return error(engine_err_toofast);
}

#define toofast_error(p) _toofast_error(p, __LINE__, MX_FUNCTION)


/**
 * @brief Update the particle velocities and positions, re-shuffle if
 *      appropriate.
 * @param e The #engine on which to run.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_advance ( struct engine *e ) {
    if(e->integrator == EngineIntegrator::FORWARD_EULER) {
        return engine_advance_forward_euler(e);
    }
    else {
        return engine_advance_runge_kutta_4(e);
    }
}

MxQuaternionf integrate_angular_velocity_exact_1(const MxVector3f& em, double deltaTime)
{
    MxVector3f ha = em * deltaTime * 0.5; // vector of half angle
    double len = ha.length(); // magnitude
    if (len > 0) {
        ha *= std::sin(len) / len;
        double w = std::cos(len);
        return MxQuaternionf(ha, w);
    } else {
        return MxQuaternionf(ha, 1.0);
    }
}

static MxQuaternionf integrate_angular_velocity_2(const MxVector3f &av, double dt) {
    float len = av.length();
    double theta = len * dt * 0.5;
    if (len > 1.0e-12) {
        double w = std::cos(theta);
        double s = std::sin(theta) / len;
        return  MxQuaternionf(av * s, w);
    } else {
        return MxQuaternionf({0.f, 0.f, 0.f}, 1.f);
    }
}

static inline void bodies_advance_forward_euler(const float dt, int cid)
{
    if(cid == 0) {
        for(int i = 0; i < _Engine.s.cuboids.size(); i++) {
            MxCuboid &c = _Engine.s.cuboids[i];
            c.orientation = c.orientation * integrate_angular_velocity_2(c.spin, dt);
            MxCuboid_UpdateAABB(&c);
        }
    }
}


static int *cell_staggered_ids(space *s) { 
    int ind = 0;
    int *ids = (int*)malloc(sizeof(int) * s->nr_real);
    for(int ii = 0; ii < 3; ii++) 
        for(int jj = 0; jj < 3; jj++) 
            for(int kk = 0; kk < 3; kk++) 
                for(int i = ii; i < s->cdim[0]; i += 3) 
                    for(int j = jj; j < s->cdim[1]; j += 3) 
                        for(int k = kk; k < s->cdim[2]; k += 3) {
                            int cid = space_cellid(s, i, j, k);
                            if(!(s->cells[cid].flags & cell_flag_ghost)) 
                                ids[ind++] = cid;
                        }
    return ids;
}

// FPTYPE dt, h[3], h2[3], maxv[3], maxv2[3], maxx[3], maxx2[3]; // h, h2: edge length of space cells.

static inline void cell_advance_forward_euler(const float dt, const float h[3], const float h2[3],
                   const float maxv[3], const float maxv2[3], const float maxx[3],
                   const float maxx2[3], int cid)
{
    space *s = &_Engine.s;
    int pid = 0;
    
    struct space_cell *c = &(s->cells[ cid ]);
    int cdim[] = {s->cdim[0], s->cdim[1], s->cdim[2]};
    FPTYPE computed_volume = 0;
    
    while ( pid < c->count ) {
        MxParticle *p = &( c->parts[pid] );
        
        if(p->flags & PARTICLE_CLUSTER || (
                                           (p->flags & PARTICLE_FROZEN_X) &&
                                           (p->flags & PARTICLE_FROZEN_Y) &&
                                           (p->flags & PARTICLE_FROZEN_Z)
                                           )) {
            pid++;
            continue;
        }
        
        float mask[] = {
            (p->flags & PARTICLE_FROZEN_X) ? 0.0f : 1.0f,
            (p->flags & PARTICLE_FROZEN_Y) ? 0.0f : 1.0f,
            (p->flags & PARTICLE_FROZEN_Z) ? 0.0f : 1.0f
        };
        
        int delta[3];
        if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
            for ( int k = 0 ; k < 3 ; k++ ) {
                float v = mask[k] * (p->v[k] + dt * p->f[k] * p->imass);
                p->v[k] = v * v <= maxv2[k] ? v : v / abs(v) * maxv[k];
                p->x[k] += dt * p->v[k];
                delta[k] = __builtin_isgreaterequal( p->x[k] , h[k] ) - __builtin_isless( p->x[k] , 0.0 );
            }
        }
        else {
            for ( int k = 0 ; k < 3 ; k++ ) {
                float dx = mask[k] * (dt * p->f[k] * p->imass);
                dx = dx * dx <= maxx2[k] ? dx : dx / abs(dx) * maxx[k];
                p->v[k] = dx / dt;
                p->x[k] += dx;
                delta[k] = __builtin_isgreaterequal( p->x[k] , h[k] ) - __builtin_isless( p->x[k] , 0.0 );
            }
        }
        
        p->inv_number_density = p->number_density > 0.f ? 1.f / p->number_density : 0.f;
        
        /* do we have to move this particle? */
        // TODO: consolidate moving to one method.
        
        // if delta is non-zero, need to check boundary conditions, and
        // if moved out of cell, or out of bounds.
        if ( ( delta[0] != 0 ) || ( delta[1] != 0 ) || ( delta[2] != 0 ) ) {
            
            // if we enforce boundary, reflect back into same cell
            if(apply_update_pos_vel(p, c, h, delta)) {
                pid += 1;
            }
            // otherwise queue move to different cell
            else {
                for ( int k = 0 ; k < 3 ; k++ ) {
                    if(p->x[k] >= h2[k] || p->x[k] <= -h[k]) {
                        toofast_error(p);
                        break;
                    }
                    float dx = - delta[k] * h[k];
                    p->x[k] += dx;
                    p->p0[k] += dx;
                }
                struct space_cell *c_dest = &( s->cells[ celldims_cellid( cdim ,
                                                   (c->loc[0] + delta[0] + cdim[0]) % cdim[0] ,
                                                   (c->loc[1] + delta[1] + cdim[1]) % cdim[1] ,
                                                   (c->loc[2] + delta[2] + cdim[2]) % cdim[2] ) ] );
                
                // update any state variables on the object accordign to the boundary conditions
                // since we might be moving across periodic boundaries.
                apply_boundary_particle_crossing(p, delta, s->celllist[ p->id ], c_dest);
                
                pthread_mutex_lock(&c_dest->cell_mutex);
                space_cell_add_incomming( c_dest , p );
                c_dest->computed_volume += p->inv_number_density;
                pthread_mutex_unlock(&c_dest->cell_mutex);
                
                s->celllist[ p->id ] = c_dest;
                
                // remove a particle from a cell. if the part was the last in the
                // cell, simply dec the count, otherwise, move the last part
                // in the cell to the ejected part's prev loc.
                c->count -= 1;
                if ( pid < c->count ) {
                    c->parts[pid] = c->parts[c->count];
                    s->partlist[ c->parts[pid].id ] = &( c->parts[pid] );
                }
            }
        }
        else {
            computed_volume += p->inv_number_density;
            pid += 1;
        }
        
        assert(p->verify());
    }

    c->computed_volume = computed_volume;
}

static inline void cell_advance_forward_euler_cluster(const float h[3], int cid) 
{
    space *s = &_Engine.s;
    space_cell *c = &(s->cells[ s->cid_real[cid] ]);
    int pid = 0;
    
    while ( pid < c->count ) {
        MxParticle *p = &( c->parts[pid] );
        if((p->flags & PARTICLE_CLUSTER) && p->nr_parts > 0) {
            
            MxCluster_ComputeAggregateQuantities((MxCluster*)p);
            
            int delta[3];
            for ( int k = 0 ; k < 3 ; k++ ) {
                delta[k] = __builtin_isgreaterequal( p->x[k] , h[k] ) - __builtin_isless( p->x[k] , 0.0 );
            }
            
            /* do we have to move this particle? */
            // TODO: consolidate moving to one method.
            if ( ( delta[0] != 0 ) || ( delta[1] != 0 ) || ( delta[2] != 0 ) ) {
                for ( int k = 0 ; k < 3 ; k++ ) {
                    p->x[k] -= delta[k] * h[k];
                    p->p0[k] -= delta[k] * h[k];
                }
                
                space_cell *c_dest = &( s->cells[ space_cellid( s ,
                                                    (c->loc[0] + delta[0] + s->cdim[0]) % s->cdim[0] ,
                                                    (c->loc[1] + delta[1] + s->cdim[1]) % s->cdim[1] ,
                                                    (c->loc[2] + delta[2] + s->cdim[2]) % s->cdim[2] ) ] );
                
                pthread_mutex_lock(&c_dest->cell_mutex);
                space_cell_add_incomming( c_dest , p );
                pthread_mutex_unlock(&c_dest->cell_mutex);
                
                s->celllist[ p->id ] = c_dest;
                
                // remove a particle from a cell. if the part was the last in the
                // cell, simply dec the count, otherwise, move the last part
                // in the cell to the ejected part's prev loc.
                c->count -= 1;
                if ( pid < c->count ) {
                    c->parts[pid] = c->parts[c->count];
                    s->partlist[ c->parts[pid].id ] = &( c->parts[pid] );
                }
            }
            else {
                pid += 1;
            }
        }
        else {
            pid += 1;
        }
    }
}

/**
 * @brief Update the particle velocities and positions, re-shuffle if
 *      appropriate.
 * @param e The #engine on which to run.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
int engine_advance_forward_euler ( struct engine *e ) {

    // set the integrator flag to set any persistent forces
    // forward euler is a single step, so alwasy set this flag
    e->integrator_flags |= INTEGRATOR_UPDATE_PERSISTENTFORCE;

    if (engine_force( e ) < 0 ) {
        Log(LOG_CRITICAL);
        return error(engine_err);
    }

    ticks tic = getticks();

    int cid, pid, k, delta[3];
    struct space_cell *c, *c_dest;
    struct MxParticle *p;
    struct space *s;
    FPTYPE dt, time, h[3], h2[3], maxv[3], maxv2[3], maxx[3], maxx2[3]; // h, h2: edge length of space cells.
    double epot = 0.0, computed_volume = 0.0;

#ifdef HAVE_OPENMP

    int step;
    double epot_local;

#endif

    /* Get a grip on the space. */
    s = &(e->s);
    time = e->time;
    dt = e->dt;
    for ( k = 0 ; k < 3 ; k++ ) {
        h[k] = s->h[k];
        h2[k] = 2. * s->h[k];
        
        maxx[k] = h[k] * e->particle_max_dist_fraction;
        maxx2[k] = maxx[k] * maxx[k];
        
        // max velocity and step, as a fraction of cell size.
        maxv[k] = maxx[k] / dt;
        maxv2[k] = maxv[k] * maxv[k];
    }

    /* update the particle velocities and positions */
    if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi)) {

        /* Collect potential energy from ghosts. */
        for ( cid = 0 ; cid < s->nr_ghost ; cid++ )
            epot += s->cells[ s->cid_ghost[cid] ].epot;

#ifdef HAVE_OPENMP
#pragma omp parallel private(cid,c,pid,p,w,k,epot_local)
        {
            step = omp_get_num_threads();
            epot_local = 0.0;
            for ( cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step ) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                for ( pid = 0 ; pid < c->count ; pid++ ) {
                    p = &( c->parts[pid] );

                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        for ( k = 0 ; k < 3 ; k++ ) {
                            p->v[k] += p->f[k] * dt * p->imass;
                            p->x[k] += p->v[k] * dt;
                        }
                    }
                    else {
                        for ( k = 0 ; k < 3 ; k++ ) {
                            p->v[k] = p->f[k] * p->imass;
                            p->x[k] += p->v[k] * dt;
                        }
                    }
                }
            }
#pragma omp atomic
            epot += epot_local;
        }
#else
        auto func_update_parts = [&](int _cid) -> void {
            space_cell *_c = &(s->cells[ s->cid_real[_cid] ]);
            for ( int _pid = 0 ; _pid < _c->count ; _pid++ ) {
                MxParticle *_p = &( _c->parts[_pid] );

                if(engine::types[_p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                    for ( int _k = 0 ; _k < 3 ; _k++ ) {
                        _p->v[_k] += _p->f[_k] * dt * _p->imass;
                        _p->x[_k] += _p->v[_k] * dt;
                    }
                }
                else {
                    for ( int _k = 0 ; _k < 3 ; _k++ ) {
                        _p->v[_k] = _p->f[_k] * _p->imass;
                        _p->x[_k] += _p->v[_k] * dt;
                    }
                }
            }
        };
        mx::parallel_for(s->nr_real, func_update_parts);
        for(cid = 0; cid < s->nr_real; cid++) {
            epot += s->cells[ s->cid_real[cid] ].epot;
            computed_volume += s->cells[s->cid_real[cid]].computed_volume;
        }
#endif
    }
    else { // NOT if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi)) {
        
        // make a lambda function that we run in parallel, capture local vars.
        // we use the same lambda in both parallel and serial versions to
        // make sure same code gets exercized.
        //
        // cell_advance_forward_euler(const float dt, const float h[3], const float h2[3],
        // const float maxv[3], const float maxv2[3], const float maxx[3],
        // const float maxx2[3], float *total_pot, int cid)

        static int *staggered_ids = cell_staggered_ids(s);
        
        auto func = [dt, &h, &h2, &maxv, &maxv2, &maxx, &maxx2](int cid) -> void {
            int _cid = staggered_ids[cid];
            cell_advance_forward_euler(dt, h, h2, maxv, maxv2, maxx, maxx2, _cid);
            
            // Patching a strange bug here. 
            // When built with CUDA support, space_cell alignment is off in MxFluxes_integrate when retrieved from static engine. 
            // TODO: fix space cell alignment issue when built with CUDA
            #ifdef HAVE_CUDA
            MxFluxes_integrate(&_Engine.s.cells[_cid], _Engine.dt);
            #else
            MxFluxes_integrate(_cid);
            #endif
            
            bodies_advance_forward_euler(dt, _cid);
        };
        
        mx::parallel_for(s->nr_real, func);

        auto func_advance_clusters = [&h](int _cid) -> void {
            cell_advance_forward_euler_cluster(h, staggered_ids[_cid]);
        };
        mx::parallel_for(s->nr_real, func_advance_clusters);

        auto func_space_cell_welcome = [&](int _cid) -> void {
            space_cell_welcome( &(s->cells[ s->cid_marked[_cid] ]) , s->partlist );
        };
        mx::parallel_for(s->nr_marked, func_space_cell_welcome);

        /* Collect potential energy and computed volume */
        for(cid = 0; cid < s->nr_cells; cid++) {
            epot += s->cells[cid].epot;
            computed_volume += s->cells[cid].computed_volume;
        }

        Log(LOG_TRACE) << "step: " << time  << ", computed volume: " << computed_volume;
    } // endif NOT if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi))

    /* Store the accumulated potential energy. */
    s->epot += epot;
    s->epot_nonbond += epot;
    e->computed_volume = computed_volume;

    VERIFY_PARTICLES();

    e->timers[engine_timer_advance] += getticks() - tic;

    /* return quietly */
    return engine_err_ok;
}

#define CHECK_TOOFAST(p, h, h2) \
{\
    for(int _k = 0; _k < 3; _k++) {\
        if (p->x[_k] >= h2[_k] || p->x[_k] <= -h[_k]) {\
            return toofast_error(p);\
        }\
    }\
}\



/**
 * @brief Update the particle velocities and positions, re-shuffle if
 *      appropriate.
 * @param e The #engine on which to run.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_advance_runge_kutta_4 ( struct engine *e ) {

    int cid, pid, k, delta[3], step;
    struct space_cell *c, *c_dest;
    struct MxParticle *p;
    struct space *s;
    FPTYPE dt, w, h[3], h2[3], maxv[3], maxv2[3], maxx[3], maxx2[3]; // h, h2: edge length of space cells.
    double epot = 0.0, epot_local;
    int toofast;

    /* Get a grip on the space. */
    s = &(e->s);
    dt = e->dt;
    for ( k = 0 ; k < 3 ; k++ ) {
        h[k] = s->h[k];
        h2[k] = 2. * s->h[k];

        maxv[k] = h[k] / (e->particle_max_dist_fraction * dt);
        maxv2[k] = maxv[k] * maxv[k];

        maxx[k] = h[k] / (e->particle_max_dist_fraction);
        maxx2[k] = maxx[k] * maxx[k];
    }

    /* update the particle velocities and positions */
    if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi)) {

        /* Collect potential energy from ghosts. */
        for ( cid = 0 ; cid < s->nr_ghost ; cid++ )
            epot += s->cells[ s->cid_ghost[cid] ].epot;

#pragma omp parallel private(cid,c,pid,p,w,k,epot_local)
        {
            step = omp_get_num_threads();
            epot_local = 0.0;
            for ( cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step ) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                for ( pid = 0 ; pid < c->count ; pid++ ) {
                    p = &( c->parts[pid] );
                    w = dt * p->imass;

                    toofast = 0;
                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        for ( k = 0 ; k < 3 ; k++ ) {
                            p->v[k] += dt * p->f[k] * w;
                            p->x[k] += dt * p->v[k];
                            delta[k] = isgreaterequal( p->x[k] , h[k] ) - isless( p->x[k] , 0.0 );
                            toofast = toofast || (p->x[k] >= h2[k] || p->x[k] <= -h[k]);
                        }
                    }
                    else {
                        for ( k = 0 ; k < 3 ; k++ ) {
                            p->v[k] = p->f[k] * w;
                            p->x[k] += dt * p->v[k];
                            delta[k] = isgreaterequal( p->x[k] , h[k] ) - isless( p->x[k] , 0.0 );
                            toofast = toofast || (p->x[k] >= h2[k] || p->x[k] <= -h[k]);
                        }
                    }
                }
            }
#pragma omp atomic
            epot += epot_local;
        }
    }
    else { // NOT if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi))

        /* Collect potential energy from ghosts. */
        for ( cid = 0 ; cid < s->nr_ghost ; cid++ ) {
            epot += s->cells[ s->cid_ghost[cid] ].epot;
        }

        // **  get K1, calculate forces at current position **
        // set the integrator flag to set any persistent forces
        e->integrator_flags |= INTEGRATOR_UPDATE_PERSISTENTFORCE;
        if (engine_force( e ) < 0 ) {
            return error(engine_err);
        }
        e->integrator_flags &= ~INTEGRATOR_UPDATE_PERSISTENTFORCE;

#pragma omp parallel private(cid,c,pid,p,w,k,delta,c_dest,epot_local,ke)
        {
            step = omp_get_num_threads(); epot_local = 0.0;
            toofast = 0;
            for ( cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step ) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                pid = 0;
                
                while ( pid < c->count ) {
                    p = &( c->parts[pid] );
                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        p->vk[0] = p->force * p->imass;
                        p->xk[0] = p->velocity;
                    }
                    else {
                        p->xk[0] = p->force * p->imass;
                    }

                    // update position for k2
                    p->p0 = p->position;
                    p->v0 = p->velocity;
                    p->position = p->p0 + 0.5 * dt * p->xk[0];
                    CHECK_TOOFAST(p, h, h2);
                    pid += 1;
                }
            }
        }

        // ** get K2, calculate forces at x0 + 1/2 dt k1 **
        if (engine_force_prep(e) < 0 || engine_force( e ) < 0 ) {
            return error(engine_err);
        }

#pragma omp parallel private(cid,c,pid,p,w,k,delta,c_dest,epot_local,ke)
        {
            step = omp_get_num_threads(); epot_local = 0.0;
            for ( cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step ) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                pid = 0;
                while ( pid < c->count ) {
                    p = &( c->parts[pid] );

                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        p->vk[1] = p->force * p->imass;
                        p->xk[1] = p->v0 + 0.5 * dt * p->vk[0];
                    }
                    else {
                        p->xk[1] = p->force * p->imass;
                    }

                    // setup pos for next k3
                    p->position = p->p0 + 0.5 * dt * p->xk[1];
                    CHECK_TOOFAST(p, h, h2);
                    pid += 1;
                }
            }
        }

        // ** get K3, calculate forces at x0 + 1/2 dt k2 **
        if (engine_force_prep(e) < 0 || engine_force( e ) < 0 ) {
            return error(engine_err);
        }

#pragma omp parallel private(cid,c,pid,p,w,k,delta,c_dest,epot_local,ke)
        {
            step = omp_get_num_threads(); epot_local = 0.0;
            for ( cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step ) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                pid = 0;
                while ( pid < c->count ) {
                    p = &( c->parts[pid] );

                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        p->vk[2] = p->force * p->imass;
                        p->xk[2] = p->v0 + 0.5 * dt * p->vk[1];
                    }
                    else {
                        p->xk[2] = p->force * p->imass;
                    }

                    // setup pos for next k3
                    p->position = p->p0 + dt * p->xk[2];
                    CHECK_TOOFAST(p, h, h2);
                    pid += 1;
                }
            }
        }

        // ** get K4, calculate forces at x0 + dt k3, final position calculation **
        if (engine_force_prep(e) < 0 || engine_force( e ) < 0 ) {
            return error(engine_err);
        }

#pragma omp parallel private(cid,c,pid,p,w,k,delta,c_dest,epot_local,ke)
        {
            step = omp_get_num_threads(); epot_local = 0.0;
            for ( cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step ) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                pid = 0;
                while ( pid < c->count ) {
                    p = &( c->parts[pid] );
                    toofast = 0;

                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        p->vk[3] = p->imass * p->force;
                        p->xk[3] = p->v0 + dt * p->vk[2];
                        p->velocity = p->v0 + (dt/6.) * (p->vk[0] + 2*p->vk[1] + 2 * p->vk[2] + p->vk[3]);
                    }
                    else {
                        p->xk[3] = p->imass * p->force;
                    }
                    
                    p->position = p->p0 + (dt/6.) * (p->xk[0] + 2*p->xk[1] + 2 * p->xk[2] + p->xk[3]);

                    for(int k = 0; k < 3; ++k) {
                        delta[k] = __builtin_isgreaterequal( p->x[k] , h[k] ) - __builtin_isless( p->x[k] , 0.0 );
                        toofast = toofast || (p->x[k] >= h2[k] || p->x[k] <= -h[k]);
                    }

                    if(toofast) {
                        return toofast_error(p);
                    }

                    /* do we have to move this particle? */
                    if ( ( delta[0] != 0 ) || ( delta[1] != 0 ) || ( delta[2] != 0 ) ) {
                        for ( k = 0 ; k < 3 ; k++ ) {
                            p->x[k] -= delta[k] * h[k];
                        }

                        c_dest = &( s->cells[ space_cellid( s ,
                                (c->loc[0] + delta[0] + s->cdim[0]) % s->cdim[0] ,
                                (c->loc[1] + delta[1] + s->cdim[1]) % s->cdim[1] ,
                                (c->loc[2] + delta[2] + s->cdim[2]) % s->cdim[2] ) ] );

                        pthread_mutex_lock(&c_dest->cell_mutex);
                        space_cell_add_incomming( c_dest , p );
                        pthread_mutex_unlock(&c_dest->cell_mutex);

                        s->celllist[ p->id ] = c_dest;

                        // remove a particle from a cell. if the part was the last in the
                        // cell, simply dec the count, otherwise, move the last part
                        // in the cell to the ejected part's prev loc.
                        c->count -= 1;
                        if ( pid < c->count ) {
                            c->parts[pid] = c->parts[c->count];
                            s->partlist[ c->parts[pid].id ] = &( c->parts[pid] );
                        }
                    }
                    else {
                        pid += 1;
                    }
                }
            }
#pragma omp atomic
            epot += epot_local;
        }

        /* Welcome the new particles in each cell. */
#pragma omp parallel for schedule(static)
        for ( cid = 0 ; cid < s->nr_marked ; cid++ ) {
            space_cell_welcome( &(s->cells[ s->cid_marked[cid] ]) , s->partlist );
        }

    } // endif  NOT if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi))

    /* Store the accumulated potential energy. */
    s->epot_nonbond += epot;
    s->epot += epot;

    /* return quietly */
    return engine_err_ok;
}


