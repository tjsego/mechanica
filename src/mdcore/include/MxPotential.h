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


/**
 * @brief A Potential object is a compiled interpolation of a given function. The 
 * Universe applies potentials to particles to calculate the net force on them. 
 * 
 * For performance reasons, Mechanica implements potentials as 
 * interpolations, which can be much faster than evaluating the function directly. 
 * 
 * A potential can be treated just like any callable object. 
 */
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

    /**
     * @brief Creates a 12-6 Lennard-Jones potential. 
     * 
     * The Lennard Jones potential has the form:
     * 
     * @f[
     * 
     *      \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) 
     * 
     * @f]
     * 
     * @param min The smallest radius for which the potential will be constructed.
     * @param max The largest radius for which the potential will be constructed.
     * @param A The first parameter of the Lennard-Jones potential.
     * @param B The second parameter of the Lennard-Jones potential.
     * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001 * (max - min). 
     * @return MxPotential* 
     */
    static MxPotential *lennard_jones_12_6(double min, double max, double A, double B, double *tol=NULL);

    /**
     * @brief Creates a potential of the sum of a 12-6 Lennard-Jones potential and a shifted Coulomb potential. 
     * 
     * The 12-6 Lennard Jones - Coulomb potential has the form:
     * 
     * @f[
     * 
     *      \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) + q \left( \frac{1}{r} - \frac{1}{max} \right)
     * 
     * @f]
     * 
     * @param min The smallest radius for which the potential will be constructed.
     * @param max The largest radius for which the potential will be constructed.
     * @param A The first parameter of the Lennard-Jones potential.
     * @param B The second parameter of the Lennard-Jones potential.
     * @param q The charge scaling of the potential.
     * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001 * (max - min). 
     * @return MxPotential* 
     */
    static MxPotential *lennard_jones_12_6_coulomb(double min, double max, double A, double B, double q, double *tol=NULL);

    /**
     * @brief Creates a soft sphere interaction potential. 
     * 
     * The soft sphere is a generalized Lennard-Jones-type potential, but with settable exponents to create a softer interaction.
     * 
     * @param kappa 
     * @param epsilon 
     * @param r0 
     * @param eta 
     * @param min 
     * @param max 
     * @param tol 
     * @param shift 
     * @return MxPotential* 
     */
    static MxPotential *soft_sphere(double kappa, double epsilon, double r0, int eta, double *min=NULL, double *max=NULL, double *tol=NULL, bool *shift=NULL);

    /**
     * @brief Creates a real-space Ewald potential. 
     * 
     * The Ewald potential has the form:
     * 
     * @f[
     * 
     *      q \frac{\mathrm{erfc}\, ( \kappa r)}{r}
     * 
     * @f]
     * 
     * @param min The smallest radius for which the potential will be constructed.
     * @param max The largest radius for which the potential will be constructed.
     * @param q The charge scaling of the potential.
     * @param kappa The screening distance of the Ewald potential.
     * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001 * (max - min). 
     * @return MxPotential* 
     */
    static MxPotential *ewald(double min, double max, double q, double kappa, double *tol=NULL);

    /**
     * @brief Creates a Coulomb potential. 
     * 
     * The Coulomb potential has the form:
     * 
     * @f[
     * 
     *      \frac{q}{r}
     * 
     * @f]
     * 
     * @param q The charge scaling of the potential. 
     * @param min The smallest radius for which the potential will be constructed. Default is 0.01. 
     * @param max The largest radius for which the potential will be constructed. Default is 2.0. 
     * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001 * (max - min). 
     * @return MxPotential* 
     */
    static MxPotential *coulomb(double q, double *min=NULL, double *max=NULL, double *tol=NULL);

    /**
     * @brief Creates a harmonic bond potential. 
     * 
     * The harmonic potential has the form: 
     * 
     * @f[
     * 
     *      k \left( r-r_0 \right)^2
     * 
     * @f]
     * 
     * @param k The energy of the bond.
     * @param r0 The bond rest length.
     * @param min The smallest radius for which the potential will be constructed. Defaults to @f$ r_0 - r_0 / 2 @f$.
     * @param max The largest radius for which the potential will be constructed. Defaults to @f$ r_0 + r_0 /2 @f$.
     * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to @f$ 0.01 \abs(max-min) @f$.
     * @return MxPotential* 
     */
    static MxPotential *harmonic(double k, double r0, double *min=NULL, double *max=NULL, double *tol=NULL);

    /**
     * @brief Creates a linear potential. 
     * 
     * The linear potential has the form:
     * 
     * @f[
     * 
     *      k r
     * 
     * @f]
     * 
     * @param k interaction strength; represents the potential energy peak value.
     * @param min The smallest radius for which the potential will be constructed. Defaults to 0.0.
     * @param max The largest radius for which the potential will be constructed. Defaults to 10.0.
     * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001.
     * @return MxPotential* 
     */
    static MxPotential *linear(double k, double *min=NULL, double *max=NULL, double *tol=NULL);

    /**
     * @brief Creates a harmonic angle potential. 
     * 
     * The harmonic angle potential has the form: 
     * 
     * @f[
     * 
     *      k \left(\theta-\theta_{0} \right)^2
     * 
     * @f]
     * 
     * @param k The energy of the angle.
     * @param theta0 The minimum energy angle.
     * @param min The smallest angle for which the potential will be constructed. Defaults to zero. 
     * @param max The largest angle for which the potential will be constructed. Defaults to PI. 
     * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.005 * (max - min). 
     * @return MxPotential* 
     */
    static MxPotential *harmonic_angle(double k, double theta0, double *min=NULL, double *max=NULL, double *tol=NULL);

    /**
     * @brief Creates a harmonic dihedral potential. 
     * 
     * The harmonic dihedral potential has the form:
     * 
     * @f[
     * 
     *      k \left( 1 + \cos( n \arccos(r)-\delta ) \right)
     * 
     * @f]
     * 
     * @param k energy of the dihedral.
     * @param n multiplicity of the dihedral.
     * @param delta minimum energy dihedral. 
     * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001. 
     * @return MxPotential* 
     */
    static MxPotential *harmonic_dihedral(double k, int n, double delta, double *tol=NULL);

    /**
     * @brief Creates a well potential. 
     * 
     * Useful for binding a particle to a region.
     * 
     * The well potential has the form: 
     * 
     * @f[
     * 
     *      \frac{k}{\left(r_0 - r\right)^{n}}
     * 
     * @f]
     * 
     * @param k potential prefactor constant, should be decreased for larger n.
     * @param n exponent of the potential, larger n makes a sharper potential.
     * @param r0 The extents of the potential, length units. Represents the maximum extents that a two objects connected with this potential should come apart.
     * @param min The smallest radius for which the potential will be constructed. Defaults to zero.
     * @param max The largest radius for which the potential will be constructed. Defaults to r0.
     * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.01 * abs(min-max).
     * @return MxPotential* 
     */
    static MxPotential *well(double k, double n, double r0, double *min=NULL, double *max=NULL, double *tol=NULL);

    /**
     * @brief Creates a generalized Lennard-Jones potential.
     * 
     * The generalized Lennard-Jones potential has the form:
     * 
     * @f[
     * 
     *      \frac{\epsilon}{n-m} \left[ m \left( \frac{r_0}{r} \right)^n - n \left( \frac{r_0}{r} \right)^m \right]
     * 
     * @f]
     * 
     * @param e effective energy of the potential. 
     * @param m order of potential. Defaults to 3
     * @param n order of potential. Defaults to 2*m.
     * @param k mimumum of the potential. Defaults to 1.
     * @param r0 mimumum of the potential. Defaults to 1. 
     * @param min The smallest radius for which the potential will be constructed. Defaults to 0.05 * r0.
     * @param max The largest radius for which the potential will be constructed. Defaults to 5 * r0.
     * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.01.
     * @param shifted Flag for whether using a shifted potential. Defaults to true. 
     * @return MxPotential* 
     */
    static MxPotential *glj(double e, double *m=NULL, double *n=NULL, double *k=NULL, double *r0=NULL, double *min=NULL, double *max=NULL, double *tol=NULL, bool *shifted=NULL);

    /**
     * @brief Creates a Morse potential. 
     * 
     * The Morse potential has the form:
     * 
     * @f[
     * 
     *      d \left(1 - e^{ -a \left(r - r_0 \right) } \right)
     * 
     * @f]
     * 
     * @param d well depth. Defaults to 1.0.
     * @param a potential width. Defaults to 6.0.
     * @param r0 equilibrium distance. Defaults to 0.0. 
     * @param min The smallest radius for which the potential will be constructed. Defaults to 0.0001.
     * @param max The largest radius for which the potential will be constructed. Defaults to 3.0.
     * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001.
     * @return MxPotential* 
     */
    static MxPotential *morse(double *d=NULL, double *a=NULL, double *r0=NULL, double *min=NULL, double *max=NULL, double *tol=NULL);

    /**
     * @brief Creates an overlapping-sphere potential from :cite:`Osborne:2017hk`. 
     * 
     * The overlapping-sphere potential has the form: 
     * 
     * @f[
     *      \mu_{ij} s_{ij}(t) \hat{\mathbf{r}}_{ij} \log \left( 1 + \frac{||\mathbf{r}_{ij}|| - s_{ij}(t)}{s_{ij}(t)} \right) 
     *          \text{ if } ||\mathbf{r}_{ij}|| < s_{ij}(t) ,
     * @f]
     * 
     * @f[
     *      \mu_{ij}\left(||\mathbf{r}_{ij}|| - s_{ij}(t)\right) \hat{\mathbf{r}}_{ij} \exp \left( -k_c \frac{||\mathbf{r}_{ij}|| - s_{ij}(t)}{s_{ij}(t)} \right) 
     *          \text{ if } s_{ij}(t) \leq ||\mathbf{r}_{ij}|| \leq r_{max} ,
     * @f]
     * 
     * @f[
     *      0 \text{ otherwise} .
     * @f]
     * 
     * Osborne refers to @f$ \mu_{ij} @f$ as a "spring constant", this 
     * controls the size of the force, and is the potential energy peak value. 
     * @f$ \hat{\mathbf{r}}_{ij} @f$ is the unit vector from particle 
     * @f$ i @f$ center to particle @f$ j @f$ center, @f$ k_C @f$ is a 
     * parameter that defines decay of the attractive force. Larger values of 
     * @f$ k_C @f$ result in a shaper peaked attraction, and thus a shorter 
     * ranged force. @f$ s_{ij}(t) @f$ is the is the sum of the radii of the 
     * two particles.
     * 
     * @param mu interaction strength, represents the potential energy peak value. Defaults to 1.0.
     * @param kc decay strength of long range attraction. Larger values make a shorter ranged function. Defaults to 1.0.
     * @param kh Optionally add a harmonic long-range attraction, same as :meth:`glj` function. Defaults to 0.0.
     * @param r0 Optional harmonic rest length, only used if `kh` is non-zero. Defaults to 0.0.
     * @param min The smallest radius for which the potential will be constructed. Defaults to 0.001.
     * @param max The largest radius for which the potential will be constructed. Defaults to 10.0.
     * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001.
     * @return MxPotential* 
     */
    static MxPotential *overlapping_sphere(double *mu=NULL, double *kc=NULL, double *kh=NULL, double *r0=NULL, double *min=NULL, double *max=NULL, double *tol=NULL);
    
    /**
     * @brief Creates a power potential. 
     * 
     * The power potential the general form of many of the potential 
     * functions, such as :meth:`linear`, etc. power has the form:
     * 
     * @f[
     * 
     *      k (r-r_0)^{\alpha}
     * 
     * @f]
     * 
     * @param k interaction strength, represents the potential energy peak value. Defaults to 1
     * @param r0 potential rest length, zero of the potential, defaults to 0.
     * @param alpha Exponent, defaults to 1.
     * @param min minimal value potential is computed for, defaults to r0 / 2.
     * @param max cutoff distance, defaults to 3 * r0.
     * @param tol Tolerance, defaults to 0.01.
     * @return MxPotential* 
     */
    static MxPotential *power(double *k=NULL, double *r0=NULL, double *alpha=NULL, double *min=NULL, double *max=NULL, double *tol=NULL);

    /**
     * @brief Creates a Dissipative Particle Dynamics potential. 
     * 
     * The Dissipative Particle Dynamics force has the form: 
     * 
     * @f[
     * 
     *      \mathbf{F}_{ij} = \mathbf{F}^C_{ij} + \mathbf{F}^D_{ij} + \mathbf{F}^R_{ij}
     * 
     * @f]
     * 
     * The conservative force is: 
     * 
     * @f[
     * 
     *      \mathbf{F}^C_{ij} = \alpha \left(1 - \frac{r_{ij}}{r_c}\right) \mathbf{e}_{ij}
     * 
     * @f]
     * 
     * The dissapative force is:
     * 
     * @f[
     * 
     *      \mathbf{F}^D_{ij} = -\gamma \left(1 - \frac{r_{ij}}{r_c}\right)^{2}(\mathbf{e}_{ij} \cdot \mathbf{v}_{ij}) \mathbf{e}_{ij}
     * 
     * @f]
     * 
     * The random force is: 
     * 
     * @f[
     * 
     *      \mathbf{F}^R_{ij} = \sigma \left(1 - \frac{r_{ij}}{r_c}\right) \xi_{ij}\Delta t^{-1/2}\mathbf{e}_{ij}
     * 
     * @f]
     * 
     * @param alpha interaction strength of the conservative force. Defaults to 1.0. 
     * @param gamma interaction strength of dissapative force. Defaults to 1.0. 
     * @param sigma strength of random force. Defaults to 1.0. 
     * @param cutoff cutoff distance. Defaults to 1.0. 
     * @param shifted Flag for whether using a shifted potential. Defaults to false. 
     * @return MxPotential* 
     */
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
