/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 * Coypright (c) 2017 Andy Somogyi (somogyie at indiana dot edu)
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
#pragma once
#ifndef INCLUDE_POTENTIAL_H_
#define INCLUDE_POTENTIAL_H_

#include "platform.h"
#include "fptype.h"

#include <limits>
#include <utility>

/* potential error codes */
#define potential_err_ok                    0
#define potential_err_null                  -1
#define potential_err_malloc                -2
#define potential_err_bounds                -3
#define potential_err_nyi                   -4
#define potential_err_ivalsmax              -5


/* some constants */
#define potential_degree                    5
#define potential_chunk                     (potential_degree+3)
#define potential_ivalsa                    1
#define potential_ivalsb                    10
#define potential_N                         100
#define potential_align                     64
#define potential_ivalsmax                  3048

#define potential_escale                    (0.079577471545947667882)
// #define potential_escale                    1.0



/* potential flags */

enum PotentialFlags {
    POTENTIAL_NONE            = 0,
    POTENTIAL_LJ126           = 1 << 0,
    POTENTIAL_EWALD           = 1 << 1,
    POTENTIAL_COULOMB         = 1 << 2,
    POTENTIAL_SINGLE          = 1 << 3,

    /** flag defined for r^2 input */
    POTENTIAL_R2              = 1 << 4,

    /** potential defined for r input (no sqrt) */
    POTENTIAL_R               = 1 << 5,

    /** potential defined for angle */
    POTENTIAL_ANGLE           = 1 << 6,

    /** potential defined for harmonic */
    POTENTIAL_HARMONIC        = 1 << 7,

    POTENTIAL_DIHEDRAL        = 1 << 8,

    /** potential defined for switch */
    POTENTIAL_SWITCH          = 1 << 9,

    POTENTIAL_REACTIVE        = 1 << 10,

    /**
     * Scaled functions take a (r0/r)^2 argument instead of an r^2,
     * they include the rest length r0, such that r0/r yields a
     * force = 0.
     */
    POTENTIAL_SCALED          = 1 << 11,

    /**
     * potential shifted by x value,
     */
    POTENTIAL_SHIFTED         = 1 << 12,

    /**
     * potential is valid for bound particles, if un-set,
     * potential is for free particles.
     */
    POTENTIAL_BOUND           = 1 << 13,
};

enum PotentialKind {
    // standard interpolated potential kind
    POTENTIAL_KIND_POTENTIAL,
    
    //
    POTENTIAL_KIND_DPD
};


/** ID of the last error. */
CAPI_DATA(int) potential_err;

typedef void (*MxPotentialEval) ( struct MxPotential *p , struct MxParticle *,
    struct MxParticle *b, FPTYPE r2 , FPTYPE *e , FPTYPE *f );

typedef struct MxPotential* (*MxPotentialCreate) (
    struct MxPotential *partial_potential,
    struct MxParticleType *a, struct MxParticleType *b );


/** The #potential structure. */
typedef struct MxPotential {
    uint32_t kind;

    /** Flags. */
    uint32_t flags;


    /** Coefficients for the interval transform. */
    FPTYPE alpha[4];

    /** The coefficients. */
    FPTYPE *c;

    FPTYPE r0_plusone;

    /** Interval edges. */
    float a, b;

    /** potential scaling constant */
    FPTYPE mu;


    /** Nr of intervals. */
    int n;

    MxPotentialCreate create_func;

    MxPotentialEval eval;

    /**
     * pointer to what kind of potential this is.
     */
    const char* name;

    MxPotential();

    std::pair<float, float> operator()(const float &r, const float &r0=-1.0);
    float force(double r, double ri=-1.0, double rj=-1.0);

    static MxPotential *lennard_jones_12_6(double min, double max, double A, double B, double *tol=NULL);
    static MxPotential *lennard_jones_12_6_coulomb(double min, double max, double A, double B, double q, double *tol=NULL);
    static MxPotential *soft_sphere(double kappa, double epsilon, double r0, int eta, double *min=NULL, double *max=NULL, double *tol=NULL, bool *shift=NULL);
    static MxPotential *ewald(double min, double max, double q, double kappa, double *tol=NULL);
    static MxPotential *coulomb(double q, double *min=NULL, double *max=NULL, double *tol=NULL);
    static MxPotential *harmonic(double k, double r0, double *min=NULL, double *max=NULL, double *tol=NULL);
    static MxPotential *linear(double k, double *min=NULL, double *max=NULL, double *tol=NULL);
    static MxPotential *harmonic_angle(double k, double theta0, double *min=NULL, double *max=NULL, double *tol=NULL);
    static MxPotential *harmonic_dihedral(double k, int n, double delta, double *tol=NULL);
    static MxPotential *well(double k, double n, double r0, double *min=NULL, double *max=NULL, double *tol=NULL);
    static MxPotential *glj(double e, double *m=NULL, double *n=NULL, double *k=NULL, double *r0=NULL, double *min=NULL, double *max=NULL, double *tol=NULL, bool *shifted=NULL);
    static MxPotential *morse(double *d=NULL, double *a=NULL, double *r0=NULL, double *min=NULL, double *max=NULL, double *tol=NULL);
    static MxPotential *overlapping_sphere(double *mu=NULL, double *kc=NULL, double *kh=NULL, double *r0=NULL, double *min=NULL, double *max=NULL, double *tol=NULL);
    static MxPotential *power(double *k=NULL, double *r0=NULL, double *alpha=NULL, double *min=NULL, double *max=NULL, double *tol=NULL);
    static MxPotential *dpd(double *alpha=NULL, double *gamma=NULL, double *sigma=NULL, double *cutoff=NULL, bool *shifted=NULL);

    float getMin();
    float getMax();
    float getCutoff();
    std::pair<float, float> getDomain();
    int getIntervals();
    bool getBound();
    void setBound(const bool &_bound);
    FPTYPE getR0();
    void setR0(const FPTYPE &_r0);
    bool getShifted();
    void setShifted(const bool &_shifted);
    bool getRSquare();
    void setRSquare(const bool &_rSquare);

} MxPotential;


/** Fictitious null potential. */
CAPI_DATA(struct MxPotential) potential_null;


/* associated functions */
CAPI_FUNC(void) potential_clear ( struct MxPotential *p );
CAPI_FUNC(int) potential_init ( struct MxPotential *p , double (*f)( double ) ,
							   double (*fp)( double ) , double (*f6p)( double ) ,
							   FPTYPE a , FPTYPE b , FPTYPE tol );

CAPI_FUNC(int) potential_getcoeffs ( double (*f)( double ) , double (*fp)( double ) ,
									 FPTYPE *xi , int n , FPTYPE *c , FPTYPE *err );

CAPI_FUNC(double) potential_getalpha ( double (*f6p)( double ) , double a , double b );

CAPI_FUNC(struct MxPotential *) potential_create_LJ126 ( double a , double b ,
														 double A , double B , double tol );
CAPI_FUNC(struct MxPotential *) potential_create_LJ126_switch ( double a , double b ,
																double A , double B ,
																double s , double tol );
CAPI_FUNC(struct MxPotential *) potential_create_LJ126_Ewald ( double a , double b ,
															   double A , double B ,
															   double q , double kappa ,
															   double tol );
CAPI_FUNC(struct MxPotential *) potential_create_LJ126_Ewald_switch ( double a , double b ,
																	  double A , double B ,
																	  double q , double kappa ,
																	  double s , double tol );

CAPI_FUNC(struct MxPotential *) potential_create_LJ126_Coulomb ( double a , double b ,
																 double A , double B ,
																 double q , double tol );
CAPI_FUNC(struct MxPotential *) potential_create_Ewald ( double a , double b ,
														 double q , double kappa ,
														 double tol );
CAPI_FUNC(struct MxPotential *) potential_create_Coulomb ( double a , double b ,
														   double q , double tol );
CAPI_FUNC(struct MxPotential *) potential_create_harmonic ( double a , double b ,
															double K , double r0 ,
															double tol );

CAPI_FUNC(struct MxPotential *) potential_create_linear ( double a , double b ,
                                                           double k , double tol );


CAPI_FUNC(struct MxPotential *) potential_create_harmonic_angle ( double a , double b ,
																  double K , double theta0 ,
																  double tol );
CAPI_FUNC(struct MxPotential *) potential_create_harmonic_dihedral ( double K , int n ,
																	 double delta , double tol );


CAPI_FUNC(struct MxPotential *) potential_create_SS1(double k, double e, double r0, double a , double b ,double tol);

CAPI_FUNC(struct MxPotential *) potential_create_SS(int eta, double k, double e,
                                                    double r0, double a , double b , double tol, bool scale = false);

CAPI_FUNC(struct MxPotential *) potential_create_SS2(double k, double e, double r0, double a , double b ,double tol);

CAPI_FUNC(struct MxPotential *) potential_create_morse(double d, double alpha, double r0,
                                                        double min, double max, double tol);


CAPI_FUNC(struct MxPotential *) potential_create_glj(
    double e, double m, double n,
    double min, double r0, double k, double max,
    double tol, bool shifted);

/**
 * Creates a square well potential of the form:
 *
 * k/(r0 - r)^n
 *
 * @param double k
 * @param double n,
 * @param double r0 ,
 * @param double tol,
 * @param double min,
 * @param double max
 */
CAPI_FUNC(struct MxPotential *) potential_create_well ( double k , double n, double r0 ,
        double tol, double min, double max);

/* These functions are now all in potential_eval.h. */
/*
    void potential_eval ( struct potential *p , FPTYPE r2 , FPTYPE *e , FPTYPE *f );
    void potential_eval_expl ( struct potential *p , FPTYPE r2 , FPTYPE *e , FPTYPE *f );
    void potential_eval_vec_4single ( struct potential *p[4] , float *r2 , float *e , float *f );
    void potential_eval_vec_4single_r ( struct potential *p[4] , float *r_in , float *e , float *f );
    void potential_eval_vec_8single ( struct potential *p[4] , float *r2 , float *e , float *f );
    void potential_eval_vec_2double ( struct potential *p[4] , FPTYPE *r2 , FPTYPE *e , FPTYPE *f );
    void potential_eval_vec_4double ( struct potential *p[4] , FPTYPE *r2 , FPTYPE *e , FPTYPE *f );
    void potential_eval_vec_4double_r ( struct potential *p[4] , FPTYPE *r , FPTYPE *e , FPTYPE *f );
    void potential_eval_r ( struct potential *p , FPTYPE r , FPTYPE *e , FPTYPE *f );
 */

/* helper functions */
CAPI_FUNC(double) potential_LJ126 ( double r , double A , double B );
CAPI_FUNC(double) potential_LJ126_p ( double r , double A , double B );
CAPI_FUNC(double) potential_LJ126_6p ( double r , double A , double B );
CAPI_FUNC(double) potential_Ewald ( double r , double kappa );
CAPI_FUNC(double) potential_Ewald_p ( double r , double kappa );
CAPI_FUNC(double) potential_Ewald_6p ( double r , double kappa );
CAPI_FUNC(double) potential_Coulomb ( double r );
CAPI_FUNC(double) potential_Coulomb_p ( double r );
CAPI_FUNC(double) potential_Coulomb_6p ( double r );
CAPI_FUNC(double) potential_switch ( double r , double A , double B );
CAPI_FUNC(double) potential_switch_p ( double r , double A , double B );

#endif // INCLUDE_POTENTIAL_H_
