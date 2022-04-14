#include "MxCTest.h"

#include <MxCParticle.h>
#include <MxCPotential.h>
#include <MxCBind.h>


float lam = -0.5;
float mu = 1.0;
unsigned int s = 3;


double He(double r, unsigned int n) {
    if(n == 0) {
        return 1.0;
    }
    else if(n == 1) {
        return r;
    } 
    else {
        return r * He(r, n - 1) - (n - 1) * He(r, n - 2);
    }
}

unsigned int factorial(unsigned int k) {
    unsigned int result = 1;
    for(unsigned int j = 1; j <= k; j++) 
        result *= j;
    return result;
}

double dgdr(double r, unsigned int n) {
    double result = 0.0;
    for(unsigned int k = 0; k < s; k++) {
        if(2 * k >= n) {
            result += factorial(2 * k) / factorial(2 * k - n) * (lam + k) * pow(mu, k) / factorial(k) * pow(r, 2 * k);
        }
    }
    return result / pow(r, (double)n);
}

double u_n(double r, unsigned int n) {
    return pow(-1, n) * He(r, n) * lam * expf(-mu * pow(r, 2.0));
}

double f_n(double r, unsigned int n) {
    double w_n = 0.0;
    for(unsigned int j = 0; j <= n; j++) {
        w_n += factorial(n) / factorial(j) / factorial(n - j) * dgdr(r, j) * u_n(r, n - j);
    }
    return 10.0 * (u_n(r, n) + w_n / lam);
}

double f(double r) {
    return f_n(r, 0);
}


int main(int argc, char** argv) {
    struct MxSimulator_ConfigHandle config;
    struct MxUniverseConfigHandle uconfig;
    struct MxBoundaryConditionsArgsContainerHandle bargs;

    float bcvel[] = {0.0, 0.0, 0.0};
    MXCTEST_CHECK(MxCBoundaryConditionsArgsContainer_init(&bargs));
    MXCTEST_CHECK(MxCBoundaryConditionsArgsContainer_setVelocity(&bargs, "left", bcvel));
    MXCTEST_CHECK(MxCBoundaryConditionsArgsContainer_setVelocity(&bargs, "right", bcvel));

    MXCTEST_CHECK(MxCSimulator_Config_init(&config));
    MXCTEST_CHECK(MxCSimulator_Config_getUniverseConfig(&config, &uconfig));
    MXCTEST_CHECK(MxCUniverseConfig_setCutoff(&uconfig, 5.0));
    MXCTEST_CHECK(MxCUniverseConfig_setBoundaryConditions(&uconfig, &bargs));
    MXCTEST_CHECK(MxCSimulator_initC(&config, NULL, 0));

    struct MxCParticleTypeStyle WellStyleDef = MxCParticleTypeStyleDef_init();
    struct MxCParticleType WellTypeDef = MxCParticleTypeDef_init();

    WellStyleDef.visible = 0;
    WellTypeDef.frozen = 1;
    WellTypeDef.style = &WellStyleDef;

    struct MxParticleTypeHandle WellType, SmallType;
    MXCTEST_CHECK(MxCParticleType_initD(&WellType, WellTypeDef));
    MXCTEST_CHECK(MxCParticleType_registerType(&WellType));
    MXCTEST_CHECK(MxCParticleType_init(&SmallType));
    MXCTEST_CHECK(MxCParticleType_setRadius(&SmallType, 0.1));
    MXCTEST_CHECK(MxCParticleType_setFrozenY(&SmallType, 1));
    MXCTEST_CHECK(MxCParticleType_registerType(&SmallType));

    struct MxPotentialHandle pot_c;
    MXCTEST_CHECK(MxCPotential_create_custom(&pot_c, 0.0, 5.0, f, NULL, NULL, NULL, NULL));
    MXCTEST_CHECK(MxCBind_types(&pot_c, &WellType, &SmallType, 0));

    float *center = (float*)malloc(3 * sizeof(float));
    MXCTEST_CHECK(MxCUniverse_getCenter(&center));
    float *dim = (float*)malloc(3 * sizeof(float));
    MXCTEST_CHECK(MxCUniverse_getDim(&dim));

    int pid;
    MXCTEST_CHECK(MxCParticleType_createParticle(&WellType, &pid, center, NULL));

    float position[] = {0.0, center[1], center[2]};
    for(unsigned int i = 0; i < 20; i++) {
        position[0] = (float)(i + 1) / 21 * dim[0];
        MXCTEST_CHECK(MxCParticleType_createParticle(&SmallType, &pid, position, NULL));
    }

    MXCTEST_CHECK(MxCTest_runQuiet(100));

    return 0;
}