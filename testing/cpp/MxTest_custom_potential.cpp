#include "MxTest.h"
#include <Mechanica.h>

#include <limits.h>


struct WellType : MxParticleType {

    WellType() : MxParticleType(true) {
        setFrozen(true);
        style->setVisible(false);
        registerType();
    };

};

struct SmallType : MxParticleType {

    SmallType() : MxParticleType(true) {
        radius = 0.1;
        setFrozenY(true);
        registerType();
    };

};


static double eq_lam = -0.5;
static double eq_mu = 1.0;
static int eq_s = 3;


static unsigned int factorial(const unsigned int &k) {
    unsigned int result = 1;
    for(unsigned int j = 1; j <= k; j++) 
        result *= j;
    return result;
}

static double He(const double &r, const unsigned int &n) {
    if(n == 0) return 1.0;
    else if(n == 1) return r;
    else return r * He(r, n - 1) - ((double)n - 1.0) * He(r, n - 2);
}

static double dgdr(const double &_r, const unsigned int &n) {
    static double eps = std::numeric_limits<double>::epsilon();
    double r = std::max(_r, eps);
    double result = 0;
    for(int k = 1; k <= eq_s; k++) 
        if(2 * k - n >= 0) 
            result += (double)factorial(2 * k) / (double)factorial(2 * k - n) * (eq_lam + k) * std::pow(eq_mu, (double)k) / (double)factorial(k) * std::pow(r, 2.0 * k);
    return result / std::pow(r, (double)n);
}

static double u_n(const double &r, const unsigned int &n) {
    return (double)std::pow(-1, (int)n) * He(r, n) * eq_lam * exp(-eq_mu * pow(r, 2.0));
}

static double f_n(const double &r, const unsigned int &n) {
    double w_n = 0.0;
    for(unsigned int j = 0; j <= n; j++) 
        w_n += (double)factorial(n) / (double)factorial(j) / (double)factorial(n - j) * dgdr(r, j) * u_n(r, n - j);
    return 10.0 * (u_n(r, n) + w_n / eq_lam);
}

static double f(double r) {
    return f_n(r, 0);
}


int main(int argc, char const *argv[])
{
    MxBoundaryConditionsArgsContainer *bcArgs = new MxBoundaryConditionsArgsContainer();
    bcArgs->setValue("x", BOUNDARY_NO_SLIP);

    MxSimulator_Config config;
    config.setWindowless(true);
    config.universeConfig.cutoff = 5.0;
    config.universeConfig.setBoundaryConditions(bcArgs);
    MXTEST_CHECK(MxSimulator_initC(config));

    WellType *well_type = new WellType();
    SmallType *small_type = new SmallType();
    well_type = (WellType*)well_type->get();
    small_type = (SmallType*)small_type->get();

    MxPotential *pot_c = MxPotential::custom(0, 5, f, NULL, NULL);
    MXTEST_CHECK(MxBind::types(pot_c, well_type, small_type));

    // Create particles
    MxVector3f ucenter = MxUniverse::getCenter();
    MxVector3f udim = MxUniverse::get()->dim();
    MxVector3f pos;
    (*well_type)(&ucenter);
    for(int i = 0; i < 20; i++) {
        pos = MxVector3f(double(i + 1) / 21.0 * udim[0], ucenter[1], ucenter[2]);
        (*small_type)(&pos);
    }

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 100));

    return S_OK;
}
