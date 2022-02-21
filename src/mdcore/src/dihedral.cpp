/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/

/* Include configuration header */
#include "mdcore_config.h"

/* Include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <limits.h>
#include <unordered_set>

/* Include some conditional headers. */
#include "mdcore_config.h"
#ifdef __SSE__
    #include <xmmintrin.h>
#endif
#ifdef WITH_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "lock.h"
#include <MxParticle.h>
#include <MxPotential.h>
#include <../../MxUtil.h>
#include <../../MxLogger.h>
#include "potential_eval.hpp"
#include <space_cell.h>
#include "space.h"
#include "engine.h"
#include "dihedral.h"
#include <../../io/MxFIO.h>
#include <../../rendering/MxStyle.hpp>

MxStyle *MxDihedral_StylePtr = new MxStyle("gold");

/* Global variables. */
/** The ID of the last error. */
int dihedral_err = dihedral_err_ok;
unsigned int dihedral_rcount = 0;

/* the error macro. */
#define error(id)				( dihedral_err = errs_register( id , dihedral_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
const char *dihedral_err_msg[2] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered."
	};
    

/**
 * @brief Evaluate a list of dihedraled interactions
 *
 * @param b Pointer to an array of #dihedral.
 * @param N Nr of dihedrals in @c b.
 * @param e Pointer to the #engine in which these dihedrals are evaluated.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 * 
 * @return #dihedral_err_ok or <0 on error (see #dihedral_err)
 */
 
int dihedral_eval ( struct MxDihedral *d , int N , struct engine *e , double *epot_out ) {

    MxDihedral *dihedral;
    int did, pid, pjd, pkd, pld, k;
    int *loci, *locj, *lock, *locl, shift[3];
    double h[3], epot = 0.0;
    struct space *s;
    struct MxParticle *pi, *pj, *pk, *pl, **partlist;
    struct space_cell **celllist;
    struct MxPotential *pot, *pota;
    std::vector<struct MxPotential *> pots;
    FPTYPE xi[3], xj[3], xk[3], xl[3], dxi[3], dxj[3], dxl[3], cphi;
    FPTYPE wi, wj, wl, fi[3], fl[3], fic, flc;
    FPTYPE t1, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21,
        t22, t24, t26, t3, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
        t41, t42, t43, t44, t45, t46, t47, t5, t6, t7, t8, t9,
        t2, t4, t23, t25, t27, t28, t51, t52, t53, t54, t59;
    std::unordered_set<struct MxDihedral*> toDestroy;
    toDestroy.reserve(N);
#if defined(VECTORIZE)
    struct MxPotential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE], *effk[VEC_SIZE], *effl[VEC_SIZE];
    FPTYPE cphiq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE ee[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE diq[VEC_SIZE*3], djq[VEC_SIZE*3], dlq[VEC_SIZE*3];
    struct MxDihedral *dihedralq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
#else
    FPTYPE ee, eff;
#endif
    
    /* Check inputs. */
    if ( d == NULL || e == NULL )
        return error(dihedral_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    partlist = s->partlist;
    celllist = s->celllist;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* Loop over the dihedrals. */
    for ( did = 0 ; did < N ; did++ ) {

        dihedral = &d[did];
        dihedral->potential_energy = 0.0;

        if(MxDihedral_decays(dihedral)) {
            toDestroy.insert(dihedral);
            continue;
        }
    
        /* Get the particles involved. */
        pid = dihedral->i; pjd = dihedral->j; pkd = dihedral->k; pld = dihedral->l;
        if ( ( pi = partlist[ pid ] ) == NULL )
            continue;
        if ( ( pj = partlist[ pjd ] ) == NULL )
            continue;
        if ( ( pk = partlist[ pkd ] ) == NULL )
            continue;
        if ( ( pl = partlist[ pld ] ) == NULL )
            continue;
            
        /* Skip if all three are ghosts. */
        if ( ( pi->flags & PARTICLE_GHOST ) &&
             ( pj->flags & PARTICLE_GHOST ) &&
             ( pk->flags & PARTICLE_GHOST ) &&
             ( pl->flags & PARTICLE_GHOST ) )
            continue;
            
        /* Get the potential. */
        if ( ( pota = d->potential ) == NULL )
            continue;

        if(pota->kind == POTENTIAL_KIND_COMBINATION && pota->flags & POTENTIAL_SUM) {
            pots = pota->constituents();
            if(pots.size() == 0) pots = {pota};
        }
        else pots = {pota};
    
        /* Get positions relative to pj's cell. */
        loci = celllist[ pid ]->loc;
        locj = celllist[ pjd ]->loc;
        lock = celllist[ pkd ]->loc;
        locl = celllist[ pld ]->loc;
        for ( k = 0 ; k < 3 ; k++ ) {
            xj[k] = pj->x[k];
            shift[k] = loci[k] - locj[k];
            if ( shift[k] > 1 )
                shift[k] = -1;
            else if ( shift[k] < -1 )
                shift[k] = 1;
            xi[k] = pi->x[k] + h[k]*shift[k];
            shift[k] = lock[k] - locj[k];
            if ( shift[k] > 1 )
                shift[k] = -1;
            else if ( shift[k] < -1 )
                shift[k] = 1;
            xk[k] = pk->x[k] + h[k]*shift[k];
            shift[k] = locl[k] - locj[k];
            if ( shift[k] > 1 )
                shift[k] = -1;
            else if ( shift[k] < -1 )
                shift[k] = 1;
            xl[k] = pl->x[k] + h[k]*shift[k];
            }
            
        /* This is Maple-generated code, see "dihedral.maple" for details. */
        t16 = xl[2]-xk[2];
        t17 = xl[1]-xk[1];
        t18 = xl[0]-xk[0];
        t2 = t18*t18;
        t4 = t17*t17;
        t23 = t16*t16;
        t10 = t2+t4+t23;
        t19 = xk[2]-xj[2];
        t20 = xk[1]-xj[1];
        t21 = xk[0]-xj[0];
        t25 = t21*t21;
        t27 = t20*t20;
        t28 = t19*t19;
        t11 = t25+t27+t28;
        t7 = t18*t21+t17*t20+t16*t19;
        t51 = t7*t7;
        t5 = t11*t10-t51;
        t22 = xi[2]-xj[2];
        t24 = xi[1]-xj[1];
        t26 = xi[0]-xj[0];
        t52 = t26*t26;
        t53 = t24*t24;
        t54 = t22*t22;
        t12 = t52+t53+t54;
        t9 = -t26*t21-t24*t20-t22*t19;
        t59 = t9*t9;
        t6 = t12*t11-t59;
        t3 = t6*t5;
        t1 = FPTYPE_ONE/FPTYPE_SQRT(t3);
        t8 = -t26*t18-t24*t17-t22*t16;
        t47 = (t9*t7-t8*t11)*t1;
        t46 = FPTYPE_TWO*t8;
        t45 = t6*t7;
        t44 = t9*t5;
        t43 = t6*t10;
        t42 = -t9-t11;
        t41 = t22*t11;
        t40 = t24*t11;
        t39 = t26*t11;
        t38 = FPTYPE_ONE/t3*t47;
        t37 = -t7*t19+t16*t11;
        t36 = -t7*t20+t17*t11;
        t35 = -t7*t21+t18*t11;
        t34 = t9*t19+t41;
        t33 = t9*t20+t40;
        t32 = t9*t21+t39;
        t31 = t5*t38;
        t30 = t6*t38;
        t15 = xk[0]-FPTYPE_TWO*xj[0]+xi[0];
        t14 = xk[1]-FPTYPE_TWO*xj[1]+xi[1];
        t13 = xk[2]-FPTYPE_TWO*xj[2]+xi[2];
        dxi[0] = t35*t1-t32*t31;
        dxi[1] = t36*t1-t33*t31;
        dxi[2] = t37*t1-t34*t31;
        dxj[0] = (t15*t7+t21*t46+t42*t18)*t1-(-t15*t44+t18*t45+(-t39-t12*t21)*t5-t21*t43)*t38;
        dxj[1] = (t14*t7+t20*t46+t42*t17)*t1-(-t14*t44+t17*t45+(-t40-t12*t20)*t5-t20*t43)*t38;
        dxj[2] = (t13*t7+t19*t46+t42*t16)*t1-(-t13*t44+t16*t45+(-t41-t12*t19)*t5-t19*t43)*t38;
        dxl[0] = t32*t1-t35*t30;
        dxl[1] = t33*t1-t36*t30;
        dxl[2] = t34*t1-t37*t30;
        cphi = FPTYPE_FMAX( -FPTYPE_ONE , FPTYPE_FMIN( FPTYPE_ONE , t47 ) );

        for(int i = 0; i < pots.size(); i++) {
            pot = pots[i];

            /* printf( "dihedral_eval: dihedral %i is %e rad.\n" , did , cphi ); */
            if ( cphi < pot->a || cphi > pot->b ) {
                printf( "dihedral_eval: dihedral %i (%s-%s-%s-%s) out of range [%e,%e], cphi=%e.\n" ,
                    did , e->types[pi->typeId].name , e->types[pj->typeId].name ,
                    e->types[pk->typeId].name , e->types[pl->typeId].name , pot->a ,
                    pot->b , cphi );
                cphi = fmax( pot->a , fmin( pot->b , cphi ) );
                }

            if(pot->kind == POTENTIAL_KIND_BYPARTICLES) {
                std::fill(std::begin(fi), std::end(fi), 0.0);
                std::fill(std::begin(fl), std::end(fl), 0.0);
                pot->eval_byparts4(pot, pi, pj, pk, pl, cphi, &ee, fi, fl);
                for (int i = 0; i < 3; ++i) {
                    pi->f[i] += (fic = fi[i]);
                    pl->f[i] += (flc = fl[i]);
                    pj->f[i] -= fic;
                    pk->f[i] -= flc;
                }
                epot += ee;
                dihedral->potential_energy += ee;
                if(dihedral->potential_energy >= dihedral->dissociation_energy)
                    toDestroy.insert(dihedral);
            }
            else {

                #ifdef VECTORIZE
                    /* add this dihedral to the interaction queue. */
                    cphiq[icount] = cphi;
                    diq[icount*3] = dxi[0];
                    diq[icount*3+1] = dxi[1];
                    diq[icount*3+2] = dxi[2];
                    djq[icount*3] = dxj[0];
                    djq[icount*3+1] = dxj[1];
                    djq[icount*3+2] = dxj[2];
                    dlq[icount*3] = dxl[0];
                    dlq[icount*3+1] = dxl[1];
                    dlq[icount*3+2] = dxl[2];
                    effi[icount] = &pi->f[0];
                    effj[icount] = &pj->f[0];
                    effk[icount] = &pk->f[0];
                    effl[icount] = &pl->f[0];
                    potq[icount] = pot;
                    dihedralq[icount] = dihedral;
                    icount += 1;
                
                    /* evaluate the interactions if the queue is full. */
                    if ( icount == VEC_SIZE ) {

                        #if defined(FPTYPE_SINGLE)
                            #if VEC_SIZE==8
                            potential_eval_vec_8single_r( potq , cphiq , ee , eff );
                            #else
                            potential_eval_vec_4single_r( potq , cphiq , ee , eff );
                            #endif
                        #elif defined(FPTYPE_DOUBLE)
                            #if VEC_SIZE==4
                            potential_eval_vec_4double_r( potq , cphiq , ee , eff );
                            #else
                            potential_eval_vec_2double_r( potq , cphiq , ee , eff );
                            #endif
                        #endif

                        /* update the forces and the energy */
                        for ( l = 0 ; l < VEC_SIZE ; l++ ) {
                            epot += ee[l];
                            dihedralq[l]->potential_energy += ee[l];
                            for ( k = 0 ; k < 3 ; k++ ) {
                                effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                                effj[l][k] -= ( wj = eff[l] * djq[3*l+k] );
                                effl[l][k] -= ( wl = eff[l] * dlq[3*l+k] );
                                effk[l][k] += wi + wj + wl;
                                }
                            if(dihedralq[l]->potential_energy >= dihedralq[l]->dissociation_energy)
                                toDestroy.insert(dihedralq[l]);
                            }

                        /* re-set the counter. */
                        icount = 0;

                        }
                #else
                    /* evaluate the dihedral */
                    #ifdef EXPLICIT_POTENTIALS
                        potential_eval_expl( pot , cphi , &ee , &eff );
                    #else
                        potential_eval_r( pot , cphi , &ee , &eff );
                    #endif
                    
                    /* update the forces */
                    for ( k = 0 ; k < 3 ; k++ ) {
                        pi->f[k] -= ( wi = eff * dxi[k] );
                        pj->f[k] -= ( wj = eff * dxj[k] );
                        pl->f[k] -= ( wl = eff * dxl[k] );
                        pk->f[k] += wi + wj + wl;
                        }

                    /* tabulate the energy */
                    epot += ee;
                    dihedral->potential_energy += ee;
                    if(dihedral->potential_energy >= dihedral->dissociation_energy)
                        toDestroy.insert(dihedral);
                #endif

            }

        }
        
        } /* loop over dihedrals. */
        
    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if ( icount > 0 ) {
    
            /* copy the first potential to the last entries */
            for ( k = icount ; k < VEC_SIZE ; k++ ) {
                potq[k] = potq[0];
                cphiq[k] = cphiq[0];
                }
    
            /* evaluate the potentials */
            #if defined(FPTYPE_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single_r( potq , cphiq , ee , eff );
                #else
                potential_eval_vec_4single_r( potq , cphiq , ee , eff );
                #endif
            #elif defined(FPTYPE_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double_r( potq , cphiq , ee , eff );
                #else
                potential_eval_vec_2double_r( potq , cphiq , ee , eff );
                #endif
            #endif
    
            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                dihedralq[l]->potential_energy += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                    effj[l][k] -= ( wj = eff[l] * djq[3*l+k] );
                    effl[l][k] -= ( wl = eff[l] * dlq[3*l+k] );
                    effk[l][k] += wi + wj + wl;
                    }
                if(dihedralq[l]->potential_energy >= dihedralq[l]->dissociation_energy)
                    toDestroy.insert(dihedralq[l]);
                }
    
            }
    #endif
    
    /* Store the potential energy. */
    *epot_out += epot;
    
    /* We're done here. */
    return dihedral_err_ok;
    
    }


/**
 * @brief Evaluate a list of dihedraled interactions
 *
 * @param b Pointer to an array of #dihedral.
 * @param N Nr of dihedrals in @c b.
 * @param e Pointer to the #engine in which these dihedrals are evaluated.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 *
 * This function differs from #dihedral_eval in that the forces are added to
 * the array @c f instead of directly in the particle data.
 * 
 * @return #dihedral_err_ok or <0 on error (see #dihedral_err)
 */
 
int dihedral_evalf ( struct MxDihedral *d , int N , struct engine *e , FPTYPE *f , double *epot_out ) {

    MxDihedral *dihedral;
    int did, pid, pjd, pkd, pld, k;
    int *loci, *locj, *lock, *locl, shift[3];
    double h[3], epot = 0.0;
    struct space *s;
    struct MxParticle *pi, *pj, *pk, *pl, **partlist;
    struct space_cell **celllist;
    struct MxPotential *pot, *pota;
    std::vector<struct MxPotential *> pots;
    FPTYPE xi[3], xj[3], xk[3], xl[3], dxi[3], dxj[3], dxl[3], cphi;
    FPTYPE wi, wj, wl, fi[3], fl[3], fic, flc;
    FPTYPE t1, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21,
        t22, t24, t26, t3, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
        t41, t42, t43, t44, t45, t46, t47, t5, t6, t7, t8, t9,
        t2, t4, t23, t25, t27, t28, t51, t52, t53, t54, t59;
    std::unordered_set<struct MxDihedral*> toDestroy;
    toDestroy.reserve(N);
#if defined(VECTORIZE)
    struct MxPotential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE], *effk[VEC_SIZE], *effl[VEC_SIZE];
    FPTYPE cphiq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE ee[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE diq[VEC_SIZE*3], djq[VEC_SIZE*3], dlq[VEC_SIZE*3];
    struct MxDihedral *dihedralq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
#else
    FPTYPE ee, eff;
#endif
    
    /* Check inputs. */
    if ( d == NULL || e == NULL )
        return error(dihedral_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    partlist = s->partlist;
    celllist = s->celllist;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* Loop over the dihedrals. */
    for ( did = 0 ; did < N ; did++ ) {

        dihedral = &d[did];
        dihedral->potential_energy = 0.0;

        if(MxDihedral_decays(dihedral)) {
            toDestroy.insert(dihedral);
            continue;
        }
    
        /* Get the particles involved. */
        pid = dihedral->i; pjd = dihedral->j; pkd = dihedral->k; pld = dihedral->l;
        if ( ( pi = partlist[ pid] ) == NULL )
            continue;
        if ( ( pj = partlist[ pjd ] ) == NULL )
            continue;
        if ( ( pk = partlist[ pkd ] ) == NULL )
            continue;
        if ( ( pl = partlist[ pld ] ) == NULL )
            continue;
        
        /* Skip if all three are ghosts. */
        if ( ( pi->flags & PARTICLE_GHOST ) &&
             ( pj->flags & PARTICLE_GHOST ) &&
             ( pk->flags & PARTICLE_GHOST ) &&
             ( pl->flags & PARTICLE_GHOST ) )
            continue;
            
        /* Get the potential. */
        if ( ( pota = d->potential ) == NULL )
            continue;

        if(pota->kind == POTENTIAL_KIND_COMBINATION && pota->flags & POTENTIAL_SUM) {
            pots = pota->constituents();
            if(pots.size() == 0) pots = {pota};
        }
        else pots = {pota};
    
        /* Get positions relative to pj. */
        loci = celllist[ pid ]->loc;
        locj = celllist[ pjd ]->loc;
        lock = celllist[ pkd ]->loc;
        locl = celllist[ pld ]->loc;
        for ( k = 0 ; k < 3 ; k++ ) {
            xj[k] = pj->x[k];
            shift[k] = loci[k] - locj[k];
            if ( shift[k] > 1 )
                shift[k] = -1;
            else if ( shift[k] < -1 )
                shift[k] = 1;
            xi[k] = pi->x[k] + h[k]*shift[k];
            shift[k] = lock[k] - locj[k];
            if ( shift[k] > 1 )
                shift[k] = -1;
            else if ( shift[k] < -1 )
                shift[k] = 1;
            xk[k] = pk->x[k] + h[k]*shift[k];
            shift[k] = locl[k] - locj[k];
            if ( shift[k] > 1 )
                shift[k] = -1;
            else if ( shift[k] < -1 )
                shift[k] = 1;
            xl[k] = pl->x[k] + h[k]*shift[k];
            }
            
        /* This is Maple-generated code, see "dihedral.maple" for details. */
        t16 = xl[2]-xk[2];
        t17 = xl[1]-xk[1];
        t18 = xl[0]-xk[0];
        t2 = t18*t18;
        t4 = t17*t17;
        t23 = t16*t16;
        t10 = t2+t4+t23;
        t19 = xk[2]-xj[2];
        t20 = xk[1]-xj[1];
        t21 = xk[0]-xj[0];
        t25 = t21*t21;
        t27 = t20*t20;
        t28 = t19*t19;
        t11 = t25+t27+t28;
        t7 = t18*t21+t17*t20+t16*t19;
        t51 = t7*t7;
        t5 = t11*t10-t51;
        t22 = xi[2]-xj[2];
        t24 = xi[1]-xj[1];
        t26 = xi[0]-xj[0];
        t52 = t26*t26;
        t53 = t24*t24;
        t54 = t22*t22;
        t12 = t52+t53+t54;
        t9 = -t26*t21-t24*t20-t22*t19;
        t59 = t9*t9;
        t6 = t12*t11-t59;
        t3 = t6*t5;
        t1 = FPTYPE_ONE/FPTYPE_SQRT(t3);
        t8 = -t26*t18-t24*t17-t22*t16;
        t47 = (t9*t7-t8*t11)*t1;
        t46 = FPTYPE_TWO*t8;
        t45 = t6*t7;
        t44 = t9*t5;
        t43 = t6*t10;
        t42 = -t9-t11;
        t41 = t22*t11;
        t40 = t24*t11;
        t39 = t26*t11;
        t38 = FPTYPE_ONE/t3*t47;
        t37 = -t7*t19+t16*t11;
        t36 = -t7*t20+t17*t11;
        t35 = -t7*t21+t18*t11;
        t34 = t9*t19+t41;
        t33 = t9*t20+t40;
        t32 = t9*t21+t39;
        t31 = t5*t38;
        t30 = t6*t38;
        t15 = xk[0]-FPTYPE_TWO*xj[0]+xi[0];
        t14 = xk[1]-FPTYPE_TWO*xj[1]+xi[1];
        t13 = xk[2]-FPTYPE_TWO*xj[2]+xi[2];
        dxi[0] = t35*t1-t32*t31;
        dxi[1] = t36*t1-t33*t31;
        dxi[2] = t37*t1-t34*t31;
        dxj[0] = (t15*t7+t21*t46+t42*t18)*t1-(-t15*t44+t18*t45+(-t39-t12*t21)*t5-t21*t43)*t38;
        dxj[1] = (t14*t7+t20*t46+t42*t17)*t1-(-t14*t44+t17*t45+(-t40-t12*t20)*t5-t20*t43)*t38;
        dxj[2] = (t13*t7+t19*t46+t42*t16)*t1-(-t13*t44+t16*t45+(-t41-t12*t19)*t5-t19*t43)*t38;
        dxl[0] = t32*t1-t35*t30;
        dxl[1] = t33*t1-t36*t30;
        dxl[2] = t34*t1-t37*t30;
        cphi = FPTYPE_FMAX( -FPTYPE_ONE , FPTYPE_FMIN( FPTYPE_ONE , t47 ) );

        for(int i = 0; i < pots.size(); i++) {
            pot = pots[i];
        
            /* printf( "dihedral_eval: dihedral %i is %e rad.\n" , did , cphi ); */
            if ( cphi < pot->a || cphi > pot->b ) {
                printf( "dihedral_evalf: dihedral %i (%s-%s-%s-%s) out of range [%e,%e], cphi=%e.\n" ,
                    did , e->types[pi->typeId].name , e->types[pj->typeId].name ,
                    e->types[pk->typeId].name , e->types[pl->typeId].name , pot->a ,
                    pot->b , cphi );
                cphi = fmax( pot->a , fmin( pot->b , cphi ) );
                }

            if(pot->kind == POTENTIAL_KIND_BYPARTICLES) {
                std::fill(std::begin(fi), std::end(fi), 0.0);
                std::fill(std::begin(fl), std::end(fl), 0.0);
                pot->eval_byparts4(pot, pi, pj, pk, pl, cphi, &ee, fi, fl);
                for (int i = 0; i < 3; ++i) {
                    pi->f[i] += (fic = fi[i]);
                    pl->f[i] += (flc = fl[i]);
                    pj->f[i] -= fic;
                    pk->f[i] -= flc;
                }
                epot += ee;
                dihedral->potential_energy += ee;
                if(dihedral->potential_energy >= dihedral->dissociation_energy)
                    toDestroy.insert(dihedral);
            }
            else {
            
                #ifdef VECTORIZE
                    /* add this dihedral to the interaction queue. */
                    cphiq[icount] = cphi;
                    diq[icount*3] = dxi[0];
                    diq[icount*3+1] = dxi[1];
                    diq[icount*3+2] = dxi[2];
                    djq[icount*3] = dxj[0];
                    djq[icount*3+1] = dxj[1];
                    djq[icount*3+2] = dxj[2];
                    dlq[icount*3] = dxl[0];
                    dlq[icount*3+1] = dxl[1];
                    dlq[icount*3+2] = dxl[2];
                    effi[icount] = &f[ 4*pid ];
                    effj[icount] = &f[ 4*pjd ];
                    effk[icount] = &f[ 4*pkd ];
                    effl[icount] = &f[ 4*pld ];
                    potq[icount] = pot;
                    dihedralq[icount] = dihedral;
                    icount += 1;

                    /* evaluate the interactions if the queue is full. */
                    if ( icount == VEC_SIZE ) {

                        #if defined(FPTYPE_SINGLE)
                            #if VEC_SIZE==8
                            potential_eval_vec_8single_r( potq , cphiq , ee , eff );
                            #else
                            potential_eval_vec_4single_r( potq , cphiq , ee , eff );
                            #endif
                        #elif defined(FPTYPE_DOUBLE)
                            #if VEC_SIZE==4
                            potential_eval_vec_4double_r( potq , cphiq , ee , eff );
                            #else
                            potential_eval_vec_2double_r( potq , cphiq , ee , eff );
                            #endif
                        #endif

                        /* update the forces and the energy */
                        for ( l = 0 ; l < VEC_SIZE ; l++ ) {
                            epot += ee[l];
                            dihedralq[l]->potential_energy += ee[l];
                            for ( k = 0 ; k < 3 ; k++ ) {
                                effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                                effj[l][k] -= ( wj = eff[l] * djq[3*l+k] );
                                effl[l][k] -= ( wl = eff[l] * dlq[3*l+k] );
                                effk[l][k] += wi + wj + wl;
                                }
                            if(dihedralq[l]->potential_energy >= dihedralq[l]->dissociation_energy)
                                toDestroy.insert(dihedralq[l]);
                            }

                        /* re-set the counter. */
                        icount = 0;

                        }
                #else
                    /* evaluate the dihedral */
                    #ifdef EXPLICIT_POTENTIALS
                        potential_eval_expl( pot , cphi , &ee , &eff );
                    #else
                        potential_eval_r( pot , cphi , &ee , &eff );
                    #endif
                    
                    /* update the forces */
                    for ( k = 0 ; k < 3 ; k++ ) {
                        f[4*pid+k] -= ( wi = eff * dxi[k] );
                        f[4*pjd+k] -= ( wj = eff * dxj[k] );
                        f[4*pld+k] -= ( wl = eff * dxl[k] );
                        f[4*pkd+k] += wi + wj + wl;
                        }

                    /* tabulate the energy */
                    epot += ee;
                    dihedral->potential_energy += ee;
                    if(dihedral->potential_energy >= dihedral->dissociation_energy)
                        toDestroy.insert(dihedral);
                #endif

            }

        }
        
        } /* loop over dihedrals. */
        
    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if ( icount > 0 ) {
    
            /* copy the first potential to the last entries */
            for ( k = icount ; k < VEC_SIZE ; k++ ) {
                potq[k] = potq[0];
                cphiq[k] = cphiq[0];
                }
    
            /* evaluate the potentials */
            #if defined(FPTYPE_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single_r( potq , cphiq , ee , eff );
                #else
                potential_eval_vec_4single_r( potq , cphiq , ee , eff );
                #endif
            #elif defined(FPTYPE_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double_r( potq , cphiq , ee , eff );
                #else
                potential_eval_vec_2double_r( potq , cphiq , ee , eff );
                #endif
            #endif
    
            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                dihedralq[l]->potential_energy += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                    effj[l][k] -= ( wj = eff[l] * djq[3*l+k] );
                    effl[l][k] -= ( wl = eff[l] * dlq[3*l+k] );
                    effk[l][k] += wi + wj + wl;
                    }
                if(dihedralq[l]->potential_energy >= dihedralq[l]->dissociation_energy)
                    toDestroy.insert(dihedralq[l]);
                }
    
            }
    #endif
    
    /* Store the potential energy. */
    *epot_out += epot;
    
    /* We're done here. */
    return dihedral_err_ok;
    
    }

void MxDihedral::init(MxPotential *potential, 
                      MxParticleHandle *p1, 
                      MxParticleHandle *p2, 
                      MxParticleHandle *p3, 
                      MxParticleHandle *p4) 
{
    this->potential = potential;
    this->i = p1->id;
    this->j = p2->id;
    this->k = p3->id;
    this->l = p4->id;

    this->creation_time = _Engine.time;
    this->dissociation_energy = std::numeric_limits<double>::max();
    this->half_life = 0.0;

    if(!this->style) this->style = MxDihedral_StylePtr;
}

MxDihedralHandle *MxDihedral::create(MxPotential *potential, 
                                 	 MxParticleHandle *p1, 
                                 	 MxParticleHandle *p2, 
                                 	 MxParticleHandle *p3, 
                                 	 MxParticleHandle *p4) 
{
    MxDihedral *dihedral = NULL;

    int id = engine_dihedral_alloc(&_Engine, &dihedral);

    if(id < 0) {
        throw std::runtime_error("could not allocate dihedral");
        return NULL;
    }

    dihedral->init(potential, p1, p2, p3, p4);
    if(dihedral->i >= 0 && dihedral->j >= 0 && dihedral->k >= 0 && dihedral->l >= 0) {
        dihedral->flags |= DIHEDRAL_ACTIVE;
        _Engine.nr_active_dihedrals++;
    }

    MxDihedralHandle *handle = new MxDihedralHandle(id);

    return handle;
}

std::string MxDihedral::toString() {
    return mx::io::toString(*this);
}

MxDihedral *MxDihedral::fromString(const std::string &str) {
    return new MxDihedral(mx::io::fromString<MxDihedral>(str));
}

MxDihedral *MxDihedralHandle::get() {
    if(id >= _Engine.dihedrals_size) throw std::range_error("Dihedral id invalid");
    if (id < 0) return NULL;
    return &_Engine.dihedrals[this->id];
}

std::string MxDihedralHandle::str() {
    std::stringstream ss;
    auto *d = this->get();
    
    ss << "Bond(i=" << d->i << ", j=" << d->j << ", k=" << d->k << ", l=" << d->l << ")";
    
    return ss.str();
}

bool MxDihedralHandle::check() {
    return (bool)this->get();
}

HRESULT MxDihedralHandle::destroy() {
    return MxDihedral_Destroy(this->get());
}

std::vector<MxDihedralHandle*> MxDihedralHandle::items() {
    std::vector<MxDihedralHandle*> list;

    for(int i = 0; i < _Engine.nr_dihedrals; ++i)
        list.push_back(new MxDihedralHandle(i));

    return list;
}

bool MxDihedralHandle::decays() {
    return MxDihedral_decays(this->get());
}

MxParticleHandle *MxDihedralHandle::operator[](unsigned int index) {
    auto *d = this->get();
    if(!d) {
        Log(LOG_ERROR) << "Invalid dihedral handle";
        return NULL;
    }

    if(index == 0) return MxParticle_FromId(d->i)->py_particle();
    else if(index == 1) return MxParticle_FromId(d->j)->py_particle();
    else if(index == 2) return MxParticle_FromId(d->k)->py_particle();
    else if(index == 3) return MxParticle_FromId(d->l)->py_particle();
    
    mx_exp(std::range_error("Index out of range (must be 0, 1, 2 or 3)"));
    return NULL;
}

double MxDihedralHandle::getEnergy() {

    MxDihedral dihedrals[] = {*this->get()};
    FPTYPE f[] = {0.0, 0.0, 0.0};
    double epot_out = 0.0;
    dihedral_evalf(dihedrals, 1, &_Engine, f, &epot_out);
    return epot_out;
}

std::vector<int32_t> MxDihedralHandle::getParts() {
    std::vector<int32_t> result;
    MxDihedral *d = this->get();
    return std::vector<int32_t>{d->i, d->j, d->k};
}

MxPotential *MxDihedralHandle::getPotential() {
    return this->get()->potential;
}

float MxDihedralHandle::getDissociationEnergy() {
    return this->get()->dissociation_energy;
}

void MxDihedralHandle::setDissociationEnergy(const float &dissociation_energy) {
    auto *d = this->get();
    d->dissociation_energy = dissociation_energy;
}

float MxDihedralHandle::getHalfLife() {
    return this->get()->half_life;
}

void MxDihedralHandle::setHalfLife(const float &half_life) {
    auto *d = this->get();
    d->half_life = half_life;
}

MxStyle *MxDihedralHandle::getStyle() {
    return this->get()->style;
}

void MxDihedralHandle::setStyle(MxStyle *style) {
    auto *d = this->get();
    d->style = style;
}

double MxDihedralHandle::getAge() {
    return (_Engine.time - this->get()->creation_time) * _Engine.dt;
}

HRESULT MxDihedral_Destroy(MxDihedral *d) {
    if(!d) return E_FAIL;

    if(d->flags & DIHEDRAL_ACTIVE) {
        bzero(d, sizeof(MxDihedral));
        _Engine.nr_active_dihedrals -= 1;
    }

    return S_OK;
}

HRESULT MxDihedral_DestroyAll() {
    for (auto dh: MxDihedralHandle::items()) dh->destroy();
    return S_OK;
}

bool MxDihedral_decays(MxDihedral *d, std::uniform_real_distribution<double> *uniform01) {
    if(!d || d->half_life <= 0.0) return false;

    bool created = uniform01 == NULL;
    if(created) uniform01 = new std::uniform_real_distribution<double>(0.0, 1.0);

    double pr = 1.0 - std::pow(2.0, -_Engine.dt / d->half_life);
    MxRandomType &MxRandom = MxRandomEngine();
    bool result = (*uniform01)(MxRandom) < pr;

    if(created) delete uniform01;

    return result;
}


namespace mx { namespace io {

#define MXDIHEDIOTOEASY(fe, key, member) \
    fe = new MxIOElement(); \
    if(toFile(member, metaData, fe) != S_OK)  \
        return E_FAIL; \
    fe->parent = fileElement; \
    fileElement->children[key] = fe;

#define MXDIHEDIOFROMEASY(feItr, children, metaData, key, member_p) \
    feItr = children.find(key); \
    if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
        return E_FAIL;

template <>
HRESULT toFile(const MxDihedral &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    MxIOElement *fe;

    MXDIHEDIOTOEASY(fe, "flags", dataElement.flags);
    MXDIHEDIOTOEASY(fe, "i", dataElement.i);
    MXDIHEDIOTOEASY(fe, "j", dataElement.j);
    MXDIHEDIOTOEASY(fe, "k", dataElement.k);
    MXDIHEDIOTOEASY(fe, "l", dataElement.l);
    MXDIHEDIOTOEASY(fe, "id", dataElement.id);
    MXDIHEDIOTOEASY(fe, "creation_time", dataElement.creation_time);
    MXDIHEDIOTOEASY(fe, "half_life", dataElement.half_life);
    MXDIHEDIOTOEASY(fe, "dissociation_energy", dataElement.dissociation_energy);
    MXDIHEDIOTOEASY(fe, "potential_energy", dataElement.potential_energy);

    fileElement->type = "Dihedral";
    
    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxDihedral *dataElement) {

    MxIOChildMap::const_iterator feItr;

    MXDIHEDIOFROMEASY(feItr, fileElement.children, metaData, "flags", &dataElement->flags);
    MXDIHEDIOFROMEASY(feItr, fileElement.children, metaData, "i", &dataElement->i);
    MXDIHEDIOFROMEASY(feItr, fileElement.children, metaData, "j", &dataElement->j);
    MXDIHEDIOFROMEASY(feItr, fileElement.children, metaData, "k", &dataElement->k);
    MXDIHEDIOFROMEASY(feItr, fileElement.children, metaData, "l", &dataElement->l);
    MXDIHEDIOFROMEASY(feItr, fileElement.children, metaData, "id", &dataElement->id);
    MXDIHEDIOFROMEASY(feItr, fileElement.children, metaData, "creation_time", &dataElement->creation_time);
    MXDIHEDIOFROMEASY(feItr, fileElement.children, metaData, "half_life", &dataElement->half_life);
    MXDIHEDIOFROMEASY(feItr, fileElement.children, metaData, "dissociation_energy", &dataElement->dissociation_energy);
    MXDIHEDIOFROMEASY(feItr, fileElement.children, metaData, "potential_energy", &dataElement->potential_energy);

    return S_OK;
}

}};
