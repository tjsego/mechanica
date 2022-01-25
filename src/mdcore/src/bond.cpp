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
#include <iostream>
#include <sstream>
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
#include <MxPotential.h>
#include "potential_eval.hpp"
#include <space_cell.h>
#include "space.h"
#include "engine.h"
#include <bond.h>

#include <../../MxLogger.h>
#include <../../MxUtil.h>
#include <../../mx_error.h>
#include <../../io/MxFIO.h>
#include <../../rendering/MxStyle.hpp>

#ifdef HAVE_CUDA
#include "bond_cuda.h"
#endif

MxStyle *MxBond_StylePtr = new MxStyle("lime");

/* Global variables. */
/** The ID of the last error. */
int bond_err = bond_err_ok;
unsigned int bond_rcount = 0;

/* the error macro. */
#define error(id)				( bond_err = errs_register( id , bond_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
const char *bond_err_msg[2] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered."
};

/**
 * check if a type pair is in a list of pairs
 * pairs has to be a vector of pairs of types
 */
static bool pair_check(std::vector<std::pair<MxParticleType*, MxParticleType*>* > *pairs, short a_typeid, short b_typeid);

/**
 * @brief Evaluate a list of bonded interactoins
 *
 * @param b Pointer to an array of #bond.
 * @param N Nr of bonds in @c b.
 * @param e Pointer to the #engine in which these bonds are evaluated.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 * 
 * @return #bond_err_ok or <0 on error (see #bond_err)
 */
 
int bond_eval ( struct MxBond *bonds , int N , struct engine *e , double *epot_out ) {

    #ifdef HAVE_CUDA
    if(e->bonds_cuda) {
        return engine_bond_eval_cuda(bonds, N, e, epot_out);
    }
    #endif

    int bid, pid, pjd, k, *loci, *locj, shift[3], ld_pots;
    double h[3], epot = 0.0;
    struct space *s;
    struct MxParticle *pi, *pj, **partlist;
    struct space_cell **celllist;
    struct MxPotential *pot, *potb;
    std::vector<struct MxPotential *> pots;
    struct MxBond *b;
    FPTYPE ee, r2, w, f[3];
    std::unordered_set<struct MxBond*> toDestroy;
    toDestroy.reserve(N);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
#if defined(VECTORIZE)
    struct MxPotential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE dx[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE pix[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE];
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eeq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE dxq[VEC_SIZE*3];
    struct MxBond *bondq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
#else
    FPTYPE eff, dx[4], pix[4];
#endif
    
    /* Check inputs. */
    if ( bonds == NULL || e == NULL )
        return error(bond_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    partlist = s->partlist;
    celllist = s->celllist;
    ld_pots = e->max_type;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
    pix[3] = FPTYPE_ZERO;
        
    /* Loop over the bonds. */
    for ( bid = 0 ; bid < N ; bid++ ) {

        b = &bonds[bid];
        b->potential_energy = 0.0;

        if(MxBond_decays(b, &uniform01)) {
            toDestroy.insert(b);
            continue;
        }

        if(!(b->flags & BOND_ACTIVE))
            continue;
    
        /* Get the particles involved. */
        pid = b->i; pjd = b->j;
        if ( ( pi = partlist[ pid ] ) == NULL )
            continue;
        if ( ( pj = partlist[ pjd ] ) == NULL )
            continue;
        
        /* Skip if both ghosts. */
        if ( ( pi->flags & PARTICLE_GHOST ) && 
             ( pj->flags & PARTICLE_GHOST ) )
            continue;
            
        /* Get the potential. */
        potb = b->potential;
        if (!potb) {
            continue;
        }
    
        /* get the distance between both particles */
        loci = celllist[ pid ]->loc; locj = celllist[ pjd ]->loc;
        for ( k = 0 ; k < 3 ; k++ ) {
            shift[k] = loci[k] - locj[k];
            if ( shift[k] > 1 )
                shift[k] = -1;
            else if ( shift[k] < -1 )
                shift[k] = 1;
            pix[k] = pi->x[k] + h[k]*shift[k];
        }
        r2 = fptype_r2( pix , pj->x , dx );

        if(potb->kind == POTENTIAL_KIND_COMBINATION && potb->flags & POTENTIAL_SUM) {
            pots = potb->constituents();
            if(pots.size() == 0) pots = {potb};
        }
        else pots = {potb};

        for(int i = 0; i < pots.size(); i++) {
            pot = pots[i];
        
            if ( r2 < pot->a*pot->a || r2 > pot->b*pot->b ) {
                //printf( "bond_eval: bond %i (%s-%s) out of range [%e,%e], r=%e.\n" ,
                //    bid , e->types[pi->typeId].name , e->types[pj->typeId].name , pot->a , pot->b , sqrt(r2) );
                r2 = fmax( pot->a*pot->a , fmin( pot->b*pot->b , r2 ) );
            }

            if(pot->kind == POTENTIAL_KIND_BYPARTICLES) {
                std::fill(std::begin(f), std::end(f), 0.0);
                pot->eval_byparts(pot, pi, pj, dx, r2, &ee, f);
                for(int i = 0; i < 3; ++i) {
                    pi->f[i] += f[i];
                    pj->f[i] -= f[i];
                }
                epot += ee;
                b->potential_energy += ee;
                if(b->potential_energy >= b->dissociation_energy)
                    toDestroy.insert(b);
            }
            else {

                #ifdef VECTORIZE
                    /* add this bond to the interaction queue. */
                    r2q[icount] = r2;
                    dxq[icount*3] = dx[0];
                    dxq[icount*3+1] = dx[1];
                    dxq[icount*3+2] = dx[2];
                    effi[icount] = pi->f;
                    effj[icount] = pj->f;
                    potq[icount] = pot;
                    bondq[icount] = b;
                    icount += 1;

                    /* evaluate the interactions if the queue is full. */
                    if ( icount == VEC_SIZE ) {

                        #if defined(FPTYPE_SINGLE)
                            #if VEC_SIZE==8
                            potential_eval_vec_8single( potq , r2q , eeq , eff );
                            #else
                            potential_eval_vec_4single( potq , r2q , eeq , eff );
                            #endif
                        #elif defined(FPTYPE_DOUBLE)
                            #if VEC_SIZE==4
                            potential_eval_vec_4double( potq , r2q , eeq , eff );
                            #else
                            potential_eval_vec_2double( potq , r2q , eeq , eff );
                            #endif
                        #endif

                        /* update the forces and the energy */
                        for ( l = 0 ; l < VEC_SIZE ; l++ ) {
                            epot += eeq[l];
                            bondq[l]->potential_energy += eeq[l];
                            if(bondq[l]->potential_energy >= bondq[l]->dissociation_energy)
                                toDestroy.insert(bondq[l]);

                            for ( k = 0 ; k < 3 ; k++ ) {
                                w = eff[l] * dxq[l*3+k];
                                effi[l][k] -= w;
                                effj[l][k] += w;
                            }
                        }

                        /* re-set the counter. */
                        icount = 0;

                    }
                #else // NOT VECTORIZE
                    /* evaluate the bond */
                    #ifdef EXPLICIT_POTENTIALS
                        potential_eval_expl( pot , r2 , &ee , &eff );
                    #else
                        potential_eval( pot , r2 , &ee , &eff );
                    #endif
                
                    /* update the forces */
                    for ( k = 0 ; k < 3 ; k++ ) {
                        w = eff * dx[k];
                        pi->f[k] -= w;
                        pj->f[k] += w;
                    }
                    /* tabulate the energy */
                    epot += ee;
                    b->potential_energy += ee;
                    if(b->potential_energy >= b->dissociation_energy) 
                        toDestroy.insert(b);


                #endif

            }

        }

        } /* loop over bonds. */
        
    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < VEC_SIZE ; k++ ) {
                potq[k] = potq[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            #if defined(VEC_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single( potq , r2q , eeq , eff );
                #else
                potential_eval_vec_4single( potq , r2q , eeq , eff );
                #endif
            #elif defined(VEC_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double( potq , r2q , eeq , eff );
                #else
                potential_eval_vec_2double( potq , r2q , eeq , eff );
                #endif
            #endif

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += eeq[l];
                bondq[l]->potential_energy += eeq[l];
                if(bondq[l]->potential_energy >= bondq[l]->dissociation_energy)
                    toDestroy.insert(bondq[l]);

                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                    }
                }

            }
    #endif

    // Destroy every bond scheduled for destruction
    for(auto bi : toDestroy)
        MxBond_Destroy(bi);
    
    /* Store the potential energy. */
    if ( epot_out != NULL )
        *epot_out += epot;
    
    /* We're done here. */
    return bond_err_ok;
    
}



/**
 * @brief Evaluate a list of bonded interactoins
 *
 * @param bonds Pointer to an array of #bond.
 * @param N Nr of bonds in @c b.
 * @param e Pointer to the #engine in which these bonds are evaluated.
 * @param forces An array of @c FPTYPE in which to aggregate the resulting forces.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 * 
 * This function differs from #bond_eval in that the forces are added to
 * the array @c f instead of directly in the particle data.
 * 
 * @return #bond_err_ok or <0 on error (see #bond_err)
 */
 
int bond_evalf ( struct MxBond *bonds , int N , struct engine *e , FPTYPE *forces , double *epot_out ) {

    int bid, pid, pjd, k, *loci, *locj, shift[3], ld_pots;
    double h[3], epot = 0.0;
    struct space *s;
    struct MxParticle *pi, *pj, **partlist;
    struct space_cell **celllist;
    struct MxPotential *pot, *potb;
    std::vector<struct MxPotential *> pots;
    struct MxBond *b;
    FPTYPE ee, r2, w, f[3];
    std::unordered_set<struct MxBond*> toDestroy;
    toDestroy.reserve(N);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
#if defined(VECTORIZE)
    struct MxPotential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE dx[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE pix[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE];
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eeq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE dxq[VEC_SIZE*3];
    struct MxBond *bondq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
#else
    FPTYPE eff, dx[4], pix[4];
#endif
    
    /* Check inputs. */
    if ( bonds == NULL || e == NULL || forces == NULL )
        return error(bond_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    partlist = s->partlist;
    celllist = s->celllist;
    ld_pots = e->max_type;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
    pix[3] = FPTYPE_ZERO;
        
    /* Loop over the bonds. */
    for ( bid = 0 ; bid < N ; bid++ ) {
        b = &bonds[bid];
        b->potential_energy = 0.0;

        if(MxBond_decays(b, &uniform01)) {
            toDestroy.insert(b);
            continue;
        }
    
        /* Get the particles involved. */
        pid = b->i; pjd = b->j;
        if ( ( pi = partlist[ pid ] ) == NULL )
            continue;
        if ( ( pj = partlist[ pjd ] ) == NULL )
            continue;
        
        /* Skip if both ghosts. */
        if ( pi->flags & PARTICLE_GHOST && pj->flags & PARTICLE_GHOST )
            continue;
            
        /* Get the potential. */
        if ( ( pot = b->potential ) == NULL )
            continue;
    
        /* get the distance between both particles */
        loci = celllist[ pid ]->loc; locj = celllist[ pjd ]->loc;
        for ( k = 0 ; k < 3 ; k++ ) {
            shift[k] = loci[k] - locj[k];
            if ( shift[k] > 1 )
                shift[k] = -1;
            else if ( shift[k] < -1 )
                shift[k] = 1;
            pix[k] = pi->x[k] + h[k]*shift[k];
            }
        r2 = fptype_r2( pix , pj->x , dx );

        if(potb->kind == POTENTIAL_KIND_COMBINATION && potb->flags & POTENTIAL_SUM) {
            pots = potb->constituents();
            if(pots.size() == 0) pots = {potb};
        }
        else pots = {potb};

        for(int i = 0; i < pots.size(); i++) {
            pot = pots[i];

            if ( r2 < pot->a*pot->a || r2 > pot->b*pot->b ) {
                printf( "bond_evalf: bond %i (%s-%s) out of range [%e,%e], r=%e.\n" ,
                    bid , e->types[pi->typeId].name , e->types[pj->typeId].name , pot->a , pot->b , sqrt(r2) );
                r2 = fmax( pot->a*pot->a , fmin( pot->b*pot->b , r2 ) );
            }

            if(pot->kind == POTENTIAL_KIND_BYPARTICLES) {
                std::fill(std::begin(f), std::end(f), 0.0);
                pot->eval_byparts(pot, pi, pj, dx, r2, &ee, f);
                for(int i = 0; i < 3; ++i) {
                    pi->f[i] += f[i];
                    pj->f[i] -= f[i];
                }
                epot += ee;
                b->potential_energy += ee;
                if(b->potential_energy >= b->dissociation_energy)
                    toDestroy.insert(b);
            }
            else {

                #ifdef VECTORIZE
                    /* add this bond to the interaction queue. */
                    r2q[icount] = r2;
                    dxq[icount*3] = dx[0];
                    dxq[icount*3+1] = dx[1];
                    dxq[icount*3+2] = dx[2];
                    effi[icount] = &( forces[ 4*pid ] );
                    effj[icount] = &( forces[ 4*pjd ] );
                    potq[icount] = pot;
                    bondq[icount] = b;
                    icount += 1;

                    /* evaluate the interactions if the queue is full. */
                    if ( icount == VEC_SIZE ) {

                        #if defined(FPTYPE_SINGLE)
                            #if VEC_SIZE==8
                            potential_eval_vec_8single( potq , r2q , eeq , eff );
                            #else
                            potential_eval_vec_4single( potq , r2q , eeq , eff );
                            #endif
                        #elif defined(FPTYPE_DOUBLE)
                            #if VEC_SIZE==4
                            potential_eval_vec_4double( potq , r2q , eeq , eff );
                            #else
                            potential_eval_vec_2double( potq , r2q , eeq , eff );
                            #endif
                        #endif

                        /* update the forces and the energy */
                        for ( l = 0 ; l < VEC_SIZE ; l++ ) {
                            epot += eeq[l];
                            bondq[l]->potential_energy += eeq[l];
                            if(bondq[l]->potential_energy >= bondq[l]->dissociation_energy)
                                toDestroy.insert(bondq[l]);

                            for ( k = 0 ; k < 3 ; k++ ) {
                                w = eff[l] * dxq[l*3+k];
                                effi[l][k] -= w;
                                effj[l][k] += w;
                            }
                        }

                        /* re-set the counter. */
                        icount = 0;

                    }
                #else
                    /* evaluate the bond */
                    #ifdef EXPLICIT_POTENTIALS
                        potential_eval_expl( pot , r2 , &ee , &eff );
                    #else
                        potential_eval( pot , r2 , &ee , &eff );
                    #endif
                    b->potential_energy += ee;
                
                    if(b->potential_energy >= b->dissociation_energy) {
                        toDestroy.insert(b);
                    }
                    else {
                        /* update the forces */
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = eff * dx[k];
                            forces[ 4*pid + k ] -= w;
                            forces[ 4*pjd + k ] += w;
                        }
                        /* tabulate the energy */
                        epot += ee;
                    }
                #endif

            }

        }

    } /* loop over bonds. */
        
    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < VEC_SIZE ; k++ ) {
                potq[k] = potq[0];
                r2q[k] = r2q[0];
            }

            /* evaluate the potentials */
            #if defined(VEC_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single( potq , r2q , eeq , eff );
                #else
                potential_eval_vec_4single( potq , r2q , eeq , eff );
                #endif
            #elif defined(VEC_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double( potq , r2q , eeq , eff );
                #else
                potential_eval_vec_2double( potq , r2q , eeq , eff );
                #endif
            #endif

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += eeq[l];
                bondq[l]->potential_energy += eeq[l];
                if(bondq[l]->potential_energy >= bondq[l]->dissociation_energy)
                    toDestroy.insert(bondq[l]);

                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                }
            }

        }
    #endif

    // Destroy every bond scheduled for destruction
    for(auto bi : toDestroy)
        MxBond_Destroy(bi);
    
    /* Store the potential energy. */
    if ( epot_out != NULL )
        *epot_out += epot;
    
    /* We're done here. */
    return bond_err_ok;
    
}

int MxBondHandle::_init(uint32_t flags, 
                        int32_t i, 
                        int32_t j, 
                        double half_life, 
                        double bond_energy, 
                        struct MxPotential *potential) 
{
    // check whether this handle has previously been initialized and return without error if so
    if (this->id > 0 && _Engine.nr_bonds > 0) return 0;

    MxBond *bond = NULL;
    
    int result = engine_bond_alloc(&_Engine, &bond);
    
    if(result < 0) {
        throw std::runtime_error("could not allocate bond");
        return E_FAIL;
    }
    
    bond->flags = flags;
    bond->i = i;
    bond->j = j;
    bond->creation_time = _Engine.time;
    bond->half_life = half_life;
    bond->dissociation_energy = bond_energy;
    if (!bond->style) bond->style = MxBond_StylePtr;
    
    if(bond->i >= 0 && bond->j >= 0) {
        bond->flags = bond->flags | BOND_ACTIVE;
        _Engine.nr_active_bonds++;
    }

    if(potential) {
        bond->potential = potential;
    }
    
    this->id = result;

    #ifdef HAVE_CUDA
    if(_Engine.bonds_cuda) 
        if(engine_cuda_add_bond(bond) < 0) 
            return error(engine_err);
    #endif

    Log(LOG_TRACE) << "Created bond: " << this->id  << ", i: " << bond->i << ", j: " << bond->j;

    return 0;
}

MxBondHandle *MxBond::create(struct MxPotential *potential, 
                             MxParticleHandle *i, 
                             MxParticleHandle *j, 
                             double *half_life, 
                             double *bond_energy, 
                             uint32_t flags)
{
    if(!potential || !i || !j) return NULL;

    auto _half_life = half_life ? *half_life : std::numeric_limits<double>::max();
    auto _bond_energy = bond_energy ? *bond_energy : std::numeric_limits<double>::max();
    return new MxBondHandle(potential, i->id, j->id, _half_life, _bond_energy, flags);
}

std::string MxBond::toString() {
    return mx::io::toString(*this);
}

MxBond *MxBond::fromString(const std::string &str) {
    return new MxBond(mx::io::fromString<MxBond>(str));
}

MxBond *MxBondHandle::get() {
#ifdef INCLUDE_ENGINE_H_
    return &_Engine.bonds[this->id];
#else
    return NULL;
#endif
};

MxBondHandle::MxBondHandle(int id) {
    if(id >= 0 && id < _Engine.nr_bonds) this->id = id;
    else throw std::range_error("invalid id");
}

MxBondHandle::MxBondHandle(struct MxPotential *potential, int32_t i, int32_t j,
        double half_life, double bond_energy, uint32_t flags) : 
    MxBondHandle()
{
    _init(flags, i, j, half_life, bond_energy, potential);
}

int MxBondHandle::init(MxPotential *pot, 
                       MxParticleHandle *p1, 
                       MxParticleHandle *p2, 
                       const double &half_life, 
                       const double &bond_energy, 
                       uint32_t flags) 
{

    Log(LOG_DEBUG);

    try {
        return _init(flags, p1->id, p2->id, half_life, bond_energy, pot);
    }
    catch (const std::exception &e) {
        return mx_exp(e);
    }
}

std::string MxBondHandle::str() {
    std::stringstream  ss;
    MxBond *bond = &_Engine.bonds[this->id];
    
    ss << "Bond(i=" << bond->i << ", j=" << bond->j << ")";
    
    return ss.str();
}

bool MxBondHandle::check() {
    return (bool)this->get();
}

double MxBondHandle::getEnergy()
{
    Log(LOG_DEBUG);
    
    MxBond *bond = this->get();
    double energy = 0;
    
    MxBond_Energy(bond, &energy);
    
    return energy;
}

std::vector<int32_t> MxBondHandle::getParts() {
    std::vector<int32_t> result;
    MxBond *bond = get();
    if(bond && bond->flags & BOND_ACTIVE) {
        result = std::vector<int32_t>{bond->i, bond->j};
    }
    return result;
}

MxPotential *MxBondHandle::getPotential() {
    MxBond *bond = get();
    if(bond && bond->flags & BOND_ACTIVE) {
        return bond->potential;
    }
    return NULL;
}

uint32_t MxBondHandle::getId() {
    MxBond *bond = get();
    if(bond && bond->flags & BOND_ACTIVE) {
        // WARNING: need the (int) cast here to pick up the
        // correct mx::cast template specialization, won't build wiht
        // an int32_specializations, claims it's duplicate for int.
        return bond->id;
    }
    return NULL;
}

float MxBondHandle::getDissociationEnergy() {
    float result;
    MxBond *bond = get();
    if (bond) result = bond->dissociation_energy;
    return result;
}

void MxBondHandle::setDissociationEnergy(const float &dissociation_energy) {
    MxBond *bond = get();
    if (bond) bond->dissociation_energy = dissociation_energy;
}

float MxBondHandle::getHalfLife() {
    MxBond *bond = get();
    if (bond) return bond->half_life;
    return NULL;
}

void MxBondHandle::setHalfLife(const float &half_life) {
    MxBond *bond = get();
    if (bond) bond->half_life = half_life;
}

bool MxBondHandle::getActive() {
    MxBond *bond = get();
    if (bond) return (bool)(bond->flags & BOND_ACTIVE);
    return false;
}

MxStyle *MxBondHandle::getStyle() {
    MxBond *bond = get();
    if (bond) return bond->style;
    return NULL;
}

void MxBondHandle::setStyle(MxStyle *style) {
    MxBond *bond = get();
    if (bond) bond->style = style;
}

double MxBondHandle::getAge() {
    MxBond *bond = get();
    if (bond) return (_Engine.time - bond->creation_time) * _Engine.dt;
    return 0;
}

static void make_pairlist(const MxParticleList *parts,
                          float cutoff, std::vector<std::pair<MxParticleType*, MxParticleType*>* > *paircheck_list,
                          PairList& pairs) {
    int i, j;
    struct MxParticle *part_i, *part_j;
    MxVector4f dx;
    MxVector4f pix, pjx;
 
    /* get the space and cutoff */
    pix[3] = FPTYPE_ZERO;
    
    float r2;
    
    float c2 = cutoff * cutoff;
    
    // TODO: more effecient to caclulate everythign in reference frame
    // of outer particle.
    
    /* loop over all particles */
    for ( i = 1 ; i < parts->nr_parts ; i++ ) {
        
        /* get the particle */
        part_i = _Engine.s.partlist[parts->parts[i]];
        
        // global position
        double *oi = _Engine.s.celllist[part_i->id]->origin;
        pix[0] = part_i->x[0] + oi[0];
        pix[1] = part_i->x[1] + oi[1];
        pix[2] = part_i->x[2] + oi[2];
        
        /* loop over all other particles */
        for ( j = 0 ; j < i ; j++ ) {
            
            /* get the other particle */
            part_j = _Engine.s.partlist[parts->parts[j]];
            
            // global position
            double *oj = _Engine.s.celllist[part_j->id]->origin;
            pjx[0] = part_j->x[0] + oj[0];
            pjx[1] = part_j->x[1] + oj[1];
            pjx[2] = part_j->x[2] + oj[2];
            
            /* get the distance between both particles */
            r2 = fptype_r2(pix.data(), pjx.data() , dx.data());
            
            if(r2 <= c2 && pair_check(paircheck_list, part_i->typeId, part_j->typeId)) {
                pairs.push_back(Pair{part_i->id,part_j->id});
            }
        } /* loop over all other particles */
    } /* loop over all particles */
}

static bool MxBond_destroyingAll = false;

CAPI_FUNC(HRESULT) MxBond_Destroy(struct MxBond *b) {
    
    std::unique_lock<std::mutex> lock(_Engine.bonds_mutex);
    
    if(b->flags & BOND_ACTIVE) {
        #ifdef HAVE_CUDA
        if(_Engine.bonds_cuda && !MxBond_destroyingAll) 
            if(engine_cuda_finalize_bond(b->id) < 0) 
                return E_FAIL;
        #endif

        // this clears the BOND_ACTIVE flag
        bzero(b, sizeof(MxBond));
        _Engine.nr_active_bonds -= 1;
    }
    return S_OK;
}

HRESULT MxBond_DestroyAll() {
    MxBond_destroyingAll = true;

    #ifdef HAVE_CUDA
    if(_Engine.bonds_cuda) 
        if(engine_cuda_finalize_bonds_all(&_Engine) < 0) 
            return E_FAIL;
    #endif

    for(auto bh : MxBondHandle::items()) bh->destroy();

    MxBond_destroyingAll = false;
    return S_OK;
}

std::vector<MxBondHandle*>* MxBondHandle::pairwise(struct MxPotential* pot,
                                                   struct MxParticleList *parts,
                                                   const double &cutoff,
                                                   std::vector<std::pair<MxParticleType*, MxParticleType*>* > *ppairs,
                                                   const double &half_life,
                                                   const double &bond_energy,
                                                   uint32_t flags) 
{
    
    PairList pairs;
    std::vector<MxBondHandle*> *bonds = new std::vector<MxBondHandle*>();
    
    try {
        make_pairlist(parts, cutoff, ppairs, pairs);
        
        for(auto &pair : pairs) {
            auto bond = new MxBondHandle(pot, pair.i, pair.j, half_life, bond_energy, flags);
            if(!bond) {
                throw std::logic_error("failed to allocated bond");
            }
            
            bonds->push_back(bond);
        }
        
        return bonds;
    }
    catch (const std::exception &e) {
        delete bonds;
        mx_exp(e);
    }
    return NULL;
}

bool MxBond_decays(MxBond *b, std::uniform_real_distribution<double> *uniform01) {
    if(!b || b->half_life <= 0.0) return false;

    bool created = uniform01 == NULL;
    if(created) uniform01 = new std::uniform_real_distribution<double>(0.0, 1.0);

    double pr = 1.0 - std::pow(2.0, -_Engine.dt / b->half_life);
    auto MxRandom = MxRandomEngine();
    bool result = (*uniform01)(MxRandom) < pr;

    if(created) delete uniform01;

    return result;
}

HRESULT MxBondHandle::destroy()
{
    Log(LOG_DEBUG);
    
    return MxBond_Destroy(this->get());
}

std::vector<MxBondHandle*> MxBondHandle::bonds() {
    std::vector<MxBondHandle*> list;
    
    for(int i = 0; i < _Engine.nr_bonds; ++i) 
        list.push_back(new MxBondHandle(i));
    
    return list;
}

std::vector<MxBondHandle*> MxBondHandle::items() {
    return bonds();
}

bool MxBondHandle::decays() {
    return MxBond_decays(&_Engine.bonds[this->id]);
}

MxParticleHandle *MxBondHandle::operator[](unsigned int index) {
    auto *b = get();
    if(!b) {
        Log(LOG_ERROR) << "Invalid bond handle";
        return NULL;
    }

    if(index == 0) return MxParticle_FromId(b->i)->py_particle();
    else if(index == 1) return MxParticle_FromId(b->j)->py_particle();
    
    mx_exp(std::range_error("Index out of range (must be 0 or 1)"));
    return NULL;
}

HRESULT MxBond_Energy (MxBond *b, double *epot_out) {
    
    int pid, pjd, k, *loci, *locj, shift[3], ld_pots;
    double h[3], epot = 0.0;
    struct space *s;
    struct MxParticle *pi, *pj, **partlist;
    struct space_cell **celllist;
    struct MxPotential *pot;
    FPTYPE r2, w;
    FPTYPE ee, eff, dx[4], pix[4];
    
    
    /* Get local copies of some variables. */
    s = &_Engine.s;
    partlist = s->partlist;
    celllist = s->celllist;

    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
    
    pix[3] = FPTYPE_ZERO;
    
    if(!(b->flags & BOND_ACTIVE))
        return -1;
    
    /* Get the particles involved. */
    pid = b->i; pjd = b->j;
    if ( ( pi = partlist[ pid ] ) == NULL )
        return -1;
    if ( ( pj = partlist[ pjd ] ) == NULL )
        return -1;
    
    /* Skip if both ghosts. */
    if ( ( pi->flags & PARTICLE_GHOST ) &&
        ( pj->flags & PARTICLE_GHOST ) )
        return 0;
    
    /* Get the potential. */
    pot = b->potential;
    if (!pot) {
        return 0;
    }
    
    /* get the distance between both particles */
    loci = celllist[ pid ]->loc; locj = celllist[ pjd ]->loc;
    for ( k = 0 ; k < 3 ; k++ ) {
        shift[k] = loci[k] - locj[k];
        if ( shift[k] > 1 )
            shift[k] = -1;
        else if ( shift[k] < -1 )
            shift[k] = 1;
        pix[k] = pi->x[k] + h[k]*shift[k];
    }
    r2 = fptype_r2( pix , pj->x , dx );
    
    if ( r2 < pot->a*pot->a || r2 > pot->b*pot->b ) {
        //printf( "bond_eval: bond %i (%s-%s) out of range [%e,%e], r=%e.\n" ,
        //    bid , e->types[pi->typeId].name , e->types[pj->typeId].name , pot->a , pot->b , sqrt(r2) );
        r2 = fmax( pot->a*pot->a , fmin( pot->b*pot->b , r2 ) );
    }
    
    /* evaluate the bond */
    potential_eval( pot , r2 , &ee , &eff );
    
    /* update the forces */
    //for ( k = 0 ; k < 3 ; k++ ) {
    //    w = eff * dx[k];
    //    pi->f[k] -= w;
    //    pj->f[k] += w;
    //}
    
    /* tabulate the energy */
    epot += ee;
    

    /* Store the potential energy. */
    if ( epot_out != NULL )
        *epot_out += epot;
    
    /* We're done here. */
    return bond_err_ok;
}

std::vector<int32_t> MxBond_IdsForParticle(int32_t pid) {
    std::vector<int32_t> bonds;
    for (int i = 0; i < _Engine.nr_bonds; ++i) {
        MxBond *b = &_Engine.bonds[i];
        if((b->flags & BOND_ACTIVE) && (b->i == pid || b->j == pid)) {
            assert(i == b->id);
            bonds.push_back(b->id);
        }
    }
    return bonds;
}

bool pair_check(std::vector<std::pair<MxParticleType*, MxParticleType*>* > *pairs, short a_typeid, short b_typeid) {
    if(!pairs) {
        return true;
    }
    
    auto *a = &_Engine.types[a_typeid];
    auto *b = &_Engine.types[b_typeid];
    
    for (int i = 0; i < (*pairs).size(); ++i) {
        std::pair<MxParticleType*, MxParticleType*> *o = (*pairs)[i];
        if((a == std::get<0>(*o) && b == std::get<1>(*o)) ||
            (b == std::get<0>(*o) && a == std::get<1>(*o))) {
            return true;
        }
    }
    return false;
}

bool contains_bond(const std::vector<MxBondHandle*> &bonds, int a, int b) {
    for(auto h : bonds) {
        MxBond *bond = &_Engine.bonds[h->id];
        if((bond->i == a && bond->j == b) || (bond->i == b && bond->j == a)) {
            return true;
        }
    }
    return false;
}

int insert_bond(std::vector<MxBondHandle*> &bonds, int a, int b,
                MxPotential *pot, MxParticleList *parts) {
    int p1 = parts->parts[a];
    int p2 = parts->parts[b];
    if(!contains_bond(bonds, p1, p2)) {
        MxBondHandle *bond = new MxBondHandle(pot, p1, p2,
                                              std::numeric_limits<double>::max(),
                                              std::numeric_limits<double>::max(),
                                              0);
        bonds.push_back(bond);
        return 1;
    }
    return 0;
}


namespace mx { namespace io {

#define MXBONDIOTOEASY(fe, key, member) \
    fe = new MxIOElement(); \
    if(toFile(member, metaData, fe) != S_OK)  \
        return E_FAIL; \
    fe->parent = fileElement; \
    fileElement->children[key] = fe;

#define MXBONDIOFROMEASY(feItr, children, metaData, key, member_p) \
    feItr = children.find(key); \
    if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
        return E_FAIL;

template <>
HRESULT toFile(const MxBond &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    MxIOElement *fe;

    MXBONDIOTOEASY(fe, "flags", dataElement.flags);
    MXBONDIOTOEASY(fe, "i", dataElement.i);
    MXBONDIOTOEASY(fe, "j", dataElement.j);
    MXBONDIOTOEASY(fe, "id", dataElement.id);
    MXBONDIOTOEASY(fe, "creation_time", dataElement.creation_time);
    MXBONDIOTOEASY(fe, "half_life", dataElement.half_life);
    MXBONDIOTOEASY(fe, "dissociation_energy", dataElement.dissociation_energy);
    MXBONDIOTOEASY(fe, "potential_energy", dataElement.potential_energy);

    fileElement->type = "Bond";
    
    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxBond *dataElement) {

    MxIOChildMap::const_iterator feItr;

    MXBONDIOFROMEASY(feItr, fileElement.children, metaData, "flags", &dataElement->flags);
    MXBONDIOFROMEASY(feItr, fileElement.children, metaData, "i", &dataElement->i);
    MXBONDIOFROMEASY(feItr, fileElement.children, metaData, "j", &dataElement->j);
    MXBONDIOFROMEASY(feItr, fileElement.children, metaData, "id", &dataElement->id);
    MXBONDIOFROMEASY(feItr, fileElement.children, metaData, "creation_time", &dataElement->creation_time);
    MXBONDIOFROMEASY(feItr, fileElement.children, metaData, "half_life", &dataElement->half_life);
    MXBONDIOFROMEASY(feItr, fileElement.children, metaData, "dissociation_energy", &dataElement->dissociation_energy);
    MXBONDIOFROMEASY(feItr, fileElement.children, metaData, "potential_energy", &dataElement->potential_energy);

    return S_OK;
}

}};
