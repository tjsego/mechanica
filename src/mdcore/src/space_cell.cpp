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


/* include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <math.h>

#pragma clang diagnostic ignored "-Wwritable-strings"


/* macro to algin memory sizes to a multiple of cell_partalign. */
#define align_ceil(v) (((v) + (cell_partalign-1) ) & ~(cell_partalign-1))

/* include local headers */
#include "errs.h"
#include "fptype.h"
#include <MxParticle.h>
#include <space_cell.h>
#include "../../MxUtil.h"

/* the error macro. */
#define error(id)				( cell_err = errs_register( id , cell_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
const char *cell_err_msg[] = {
		"Nothing bad happened.",
		"An unexpected NULL pointer was encountered.",
		"A call to malloc failed, probably due to insufficient memory.",
		"A call to a pthread routine failed."
};


/* Map shift vector to sortlist. */
const char cell_sortlistID[27] = {
		/* ( -1 , -1 , -1 ) */   0 ,
		/* ( -1 , -1 ,  0 ) */   1 ,
		/* ( -1 , -1 ,  1 ) */   2 ,
		/* ( -1 ,  0 , -1 ) */   3 ,
		/* ( -1 ,  0 ,  0 ) */   4 ,
		/* ( -1 ,  0 ,  1 ) */   5 ,
		/* ( -1 ,  1 , -1 ) */   6 ,
		/* ( -1 ,  1 ,  0 ) */   7 ,
		/* ( -1 ,  1 ,  1 ) */   8 ,
		/* (  0 , -1 , -1 ) */   9 ,
		/* (  0 , -1 ,  0 ) */   10 ,
		/* (  0 , -1 ,  1 ) */   11 ,
		/* (  0 ,  0 , -1 ) */   12 ,
		/* (  0 ,  0 ,  0 ) */   0 ,
		/* (  0 ,  0 ,  1 ) */   12 ,
		/* (  0 ,  1 , -1 ) */   11 ,
		/* (  0 ,  1 ,  0 ) */   10 ,
		/* (  0 ,  1 ,  1 ) */   9 ,
		/* (  1 , -1 , -1 ) */   8 ,
		/* (  1 , -1 ,  0 ) */   7 ,
		/* (  1 , -1 ,  1 ) */   6 ,
		/* (  1 ,  0 , -1 ) */   5 ,
		/* (  1 ,  0 ,  0 ) */   4 ,
		/* (  1 ,  0 ,  1 ) */   3 ,
		/* (  1 ,  1 , -1 ) */   2 ,
		/* (  1 ,  1 ,  0 ) */   1 ,
		/* (  1 ,  1 ,  1 ) */   0
};
const FPTYPE cell_shift[13*3] = {
		5.773502691896258e-01 ,  5.773502691896258e-01 ,  5.773502691896258e-01 ,
		7.071067811865475e-01 ,  7.071067811865475e-01 ,  0.0                   ,
		5.773502691896258e-01 ,  5.773502691896258e-01 , -5.773502691896258e-01 ,
		7.071067811865475e-01 ,  0.0                   ,  7.071067811865475e-01 ,
		1.0                   ,  0.0                   ,  0.0                   ,
		7.071067811865475e-01 ,  0.0                   , -7.071067811865475e-01 ,
		5.773502691896258e-01 , -5.773502691896258e-01 ,  5.773502691896258e-01 ,
		7.071067811865475e-01 , -7.071067811865475e-01 ,  0.0                   ,
		5.773502691896258e-01 , -5.773502691896258e-01 , -5.773502691896258e-01 ,
		0.0                   ,  7.071067811865475e-01 ,  7.071067811865475e-01 ,
		0.0                   ,  1.0                   ,  0.0                   ,
		0.0                   ,  7.071067811865475e-01 , -7.071067811865475e-01 ,
		0.0                   ,  0.0                   ,  1.0                   ,
};
const char cell_flip[27] = { 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 ,
		0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 };


/* the last error */
int cell_err = cell_err_ok;


/**
 * @brief Flush all the parts from a #cell.
 *
 * @param c The #cell to flush.
 * @param partlist A pointer to the partlist to set the part indices.
 * @param celllist A pointer to the celllist to set the part indices.
 *
 * @return #cell_err_ok or < 0 on error (see #cell_err).
 */

int space_cell_flush ( struct space_cell *c , struct MxParticle **partlist , struct space_cell **celllist ) {

	int k;

	/* Check the inputs. */
	if ( c == NULL )
		return error(cell_err_null);

	/* Unhook the cells from the partlist. */
	if ( partlist != NULL )
		for ( k = 0 ; k < c->count ; k++ )
			partlist[ c->parts[k].id ] = NULL;

	/* Unhook the cells from the celllist. */
	if ( celllist != NULL )
		for ( k = 0 ; k < c->count ; k++ )
			celllist[ c->parts[k].id ] = NULL;

	/* Set the count to zero. */
	c->count = 0;

	/* All done! */
	return cell_err_ok;

}


/**
 * @brief Load a block of particles to the cell.
 *
 * @param c The #cell.
 * @param parts Pointer to a block of #part.
 * @param nr_parts The number of parts to load.
 * @param partlist A pointer to the partlist to set the part indices.
 * @param celllist A pointer to the celllist to set the part indices.
 *
 * @return #cell_err_ok or < 0 on error (see #cell_err).
 */

int space_cell_load ( struct space_cell *c , struct MxParticle *parts , int nr_parts , struct MxParticle **partlist , struct space_cell **celllist ) {

	int k, size_new;
	struct MxParticle *temp;

	/* check inputs */
	if ( c == NULL || parts == NULL )
		return error(cell_err_null);

	/* Is there sufficient room for these particles? */
	if ( c->count + nr_parts > c->size ) {
		size_new = c->count + nr_parts;
		if ( size_new < c->size + cell_incr )
			size_new = c->size + cell_incr;
		if ((temp = (MxParticle*)MxAligned_Malloc(align_ceil(sizeof(struct MxParticle) * size_new),  cell_partalign)) == 0 )
			return error(cell_err_malloc);
		memcpy( temp , c->parts , sizeof(struct MxParticle) * c->count );
		MxAligned_Free( c->parts );
		c->parts = temp;
		c->size = size_new;
		if ( partlist != NULL )
			for ( k = 0 ; k < c->count ; k++ )
				partlist[ c->parts[k].id ] = &( c->parts[k] );
		if ( c->sortlist != NULL ) {
			free( c->sortlist );
			if ( ( c->sortlist = (unsigned int *)malloc( sizeof(unsigned int) * 13 * c->size ) ) == NULL )
				return error(cell_err_malloc);
		}
	}

	/* Copy the new particles in. */
	memcpy( &( c->parts[c->count] ) , parts , sizeof(struct MxParticle) * nr_parts );

	/* Link them in the partlist. */
	if ( partlist != NULL )
		for ( k = c->count ; k < c->count + nr_parts ; k++ )
			partlist[ c->parts[k].id ] = &( c->parts[k] );

	/* Link them in the celllist. */
	if ( celllist != NULL )
		for ( k = c->count ; k < c->count + nr_parts ; k++ )
			celllist[ c->parts[k].id ] = c;

	/* Mark them as ghosts? */
	if ( c->flags & cell_flag_ghost )
		for ( k = c->count ; k < c->count + nr_parts ; k++ )
			c->parts[k].flags |= PARTICLE_GHOST;
	else
		for ( k = c->count ; k < c->count + nr_parts ; k++ )
			c->parts[k].flags &= ~PARTICLE_GHOST;

	/* Adjust the count. */
	c->count += nr_parts;

	/* We're out of here! */
	return cell_err_ok;

}


/**
 * @brief Move particles from the incomming buffer to the cell.
 *
 * @param c The #cell.
 * @param partlist A pointer to the partlist to set the part indices.
 *
 * @return #cell_err_ok or < 0 on error (see #cell_err).
 */

int space_cell_welcome (space_cell *c , struct MxParticle **partlist ) {

	int k;

	/* Check inputs. */
	if ( c == NULL )
		return error(cell_err_null);

	/* Loop over the incomming parts. */
	for ( k = 0 ; k < c->incomming_count ; k++ )
		if ( !space_cell_add( c , &c->incomming[k] , partlist ))
			return error(cell_err);


	/* Clear the incomming particles list. */
	c->incomming_count = 0;

	/* All done! */
	return cell_err_ok;

}


/**
 * @brief Add a particle to the incomming array of a cell.
 *
 * @param c The #cell to which the particle should be added.
 * @param p The #particle to add to the cell
 *
 * @return A pointer to the particle data in the incomming array of
 *      the cell.
 *
 * This routine assumes the particle position has already been adjusted
 * to the cell @c c.
 */

struct MxParticle *space_cell_add_incomming ( struct space_cell *c , struct MxParticle *p ) {

	struct MxParticle *temp;

	/* check inputs */
	if ( c == NULL || p == NULL ) {
		error(cell_err_null);
		return NULL;
	}

	/* is there room for this particle? */
	if ( c->incomming_count == c->incomming_size ) {
		if ((temp = (MxParticle*)MxAligned_Malloc(align_ceil(sizeof(struct MxParticle) * (c->incomming_size + cell_incr)),  cell_partalign)) == 0 ) {
			error(cell_err_malloc);
			return NULL;
		}
		memcpy( temp , c->incomming , sizeof(struct MxParticle) * c->incomming_count );
		MxAligned_Free( c->incomming );
		c->incomming = temp;
		c->incomming_size += cell_incr;
	}

	/* store this particle */
	c->incomming[c->incomming_count] = *p;

	/* all is well */
	return &( c->incomming[ c->incomming_count++ ] );

}


/**
 * @brief Add one or more particles to the incomming array of a cell.
 *
 * @param c The #cell to which the particle should be added.
 * @param p The #particle to add to the cell
 *
 * @return The number of incomming parts or < 0 on error (see #cell_err).
 *
 * This routine assumes the particle position have already been adjusted
 * to the cell @c c.
 */

int space_cell_add_incomming_multiple ( struct space_cell *c , struct MxParticle *p , int count ) {

	struct MxParticle *temp;
	int incr = cell_incr;

	/* check inputs */
	if ( c == NULL || p == NULL )
		return error(cell_err_null);

	/* is there room for this particle? */
	if ( c->incomming_count + count > c->incomming_size ) {
		if ( c->incomming_size + incr < c->incomming_count + count )
			incr = c->incomming_count + count - c->incomming_size;
		if ((temp = (MxParticle*)MxAligned_Malloc(align_ceil(sizeof(struct MxParticle) * (c->incomming_size + incr)), cell_partalign)) == 0)
			return error(cell_err_malloc);
		memcpy( temp , c->incomming , sizeof(struct MxParticle) * c->incomming_count );
		MxAligned_Free( c->incomming );
		c->incomming = temp;
		c->incomming_size += incr;
	}

	/* store this particle */
	memcpy( &c->incomming[c->incomming_count] , p , sizeof(struct MxParticle) * count );

	/* all is well */
	return ( c->incomming_count += count );

}


/**
 * @brief Add a particle to a cell.
 *
 * @param c The #cell to which the particle should be added.
 * @param p The #particle to add to the cell
 *
 * @return A pointer to the particle data in the cell.
 *
 * This routine assumes the particle position has already been adjusted
 * to the cell @c c.
 */

struct MxParticle *space_cell_add ( struct space_cell *c , struct MxParticle *p , struct MxParticle **partlist ) {

	struct MxParticle *temp;
	int k;

	/* check inputs */
	if ( c == NULL || p == NULL ) {
		error(cell_err_null);
		return NULL;
	}

	/* is there room for this particle? */
	if ( c->count == c->size ) {
		c->size *= 1.414;
		if ((temp = (MxParticle*)MxAligned_Malloc(align_ceil(sizeof(struct MxParticle) * c->size), cell_partalign)) == 0 ) {
			error(cell_err_malloc);
			return NULL;
		}
		memcpy( temp , c->parts , sizeof(struct MxParticle) * c->count );
		MxAligned_Free( c->parts );
		c->parts = temp;
		if ( partlist != NULL )
			for ( k = 0 ; k < c->count ; k++ )
				partlist[ c->parts[k].id ] = &( c->parts[k] );
		if ( c->sortlist != NULL ) {
			free( c->sortlist );
			if ( ( c->sortlist = (unsigned int *)malloc( sizeof(unsigned int) * 13 * c->size ) ) == NULL ) {
				error(cell_err_malloc);
				return NULL;
			}
		}
	}

	/* store this particle */
	c->parts[c->count] = *p;
	if ( partlist != NULL )
		partlist[ p->id ] = &c->parts[ c->count ];

	/* Mark it as a ghost? */
	if ( c->flags & cell_flag_ghost )
		c->parts[c->count].flags |= PARTICLE_GHOST;
	else
		c->parts[c->count].flags &= ~PARTICLE_GHOST;

	/* all is well */
	return &( c->parts[ c->count++ ] );

}

int space_cell_remove(struct space_cell *c, struct MxParticle *p, struct MxParticle **partlist) {
	/* check inputs */
	if ( c == NULL || p == NULL ) {
		return cell_err_null;
	}

	// index of cell in cell particle array.
    size_t cid = p - c->parts;

	assert(p == &c->parts[cid] && "pointer arithmetic error");

	c->count -= 1;
    if (cid < c->count) {
        c->parts[cid] = c->parts[c->count];
		if(partlist != NULL) {
			partlist[c->parts[cid].id] = &(c->parts[cid]);
		}
    }

	return cell_err_ok;
}


/**
 * @brief Initialize the given cell.
 *
 * @param c The #cell to initialize.
 * @param loc Array containing the location of this cell in the space.
 * @param origin The origin of the cell in global coordinates
 * @param dim The cell dimensions.
 *
 * @return #cell_err_ok or < 0 on error (see #cell_err).
 */

int space_cell_init (struct space_cell *c , int *loc , double *origin , double *dim ) {

	int i;

	/* check inputs */
	if ( c == NULL || loc == NULL || origin == NULL || dim == NULL )
		return error(cell_err_null);

	/* default flags. */
	c->flags = cell_flag_none;
	c->nodeID = 0;
	c->GPUID = 0;
    
    /*
    
    void* genAddr = &c->gen;
    void* gaussAddr = &c->gaussian;
    
    std::mt19937 *gen = new(genAddr) std::mt19937(size_t(c));
    
    // instance of class std::normal_distribution with 0 mean, and 1 stdev
    std::normal_distribution<float> *gaussian = new(gaussAddr) std::normal_distribution<float>(0.f, 1.f);
     */


	/* Init this cell's mutex. */
	if ( pthread_mutex_init( &c->cell_mutex , NULL ) != 0 )
		return error(cell_err_pthread);
	if ( pthread_cond_init( &c->cell_cond , NULL ) != 0 )
		return error(cell_err_pthread);

	/* store values */
	for ( i = 0 ; i < 3 ; i++ ) {
		c->loc[i] = loc[i];
		c->origin[i] = origin[i];
		c->dim[i] = dim[i];
	}

	/* allocate the particle pointers */
	if ((c->parts = (MxParticle*)MxAligned_Malloc(align_ceil(sizeof(struct MxParticle) * cell_default_size), cell_partalign)) == 0 )
		return error(cell_err_malloc);
	c->size = cell_default_size;
	c->count = 0;
	c->oldx_size = 0;
	c->oldx = NULL;
	if ( ( c->sortlist = (unsigned int *)malloc( sizeof(unsigned int) * 13 * c->size ) ) == NULL )
		return error(cell_err_malloc);

	/* allocate the incomming part buffer. */
	if ((c->incomming = (MxParticle*)MxAligned_Malloc(align_ceil(sizeof(struct MxParticle) * cell_incr), cell_partalign)) == 0 )
		return error(cell_err_malloc);
	c->incomming_size = cell_incr;
	c->incomming_count = 0;

	/* all is well... */
	return cell_err_ok;

}


std::ostream& operator<<(std::ostream& os, const space_cell* c) {
    os << "space_cell {" << std::endl;
    os << "\t id: " << c->id << "," << std::endl;
    os << "\t loc: [" << c->loc[0] << ", " << c->loc[1] << ", " << c->loc[2] << "]," << std::endl;
    os << "\t origin: [" << c->origin[0] << ", " << c->origin[1] << ", " << c->origin[2] << "]," << std::endl;
    os << "\t dim: [" << c->dim[0] << ", " << c->dim[1] << ", " << c->dim[2] << "]," << std::endl;
    os << "\t boundary: {" << std::endl;
    os << "\t\t left: "   << (bool)(c->flags & cell_active_left) << "," << std::endl;
    os << "\t\t right: "  << (bool)(c->flags & cell_active_right) << "," <<  std::endl;
    os << "\t\t front: "  << (bool)(c->flags & cell_active_front) << "," << std::endl;
    os << "\t\t back: "   << (bool)(c->flags & cell_active_back) << "," << std::endl;
    os << "\t\t top: "    << (bool)(c->flags & cell_active_top) << "," << std::endl;
    os << "\t\t bottom: " << (bool)(c->flags & cell_active_bottom) << "," << std::endl;
    os << "\t}" << std::endl;
    os << "}" << std::endl;
    return os;
}
