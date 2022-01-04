/**
 * @file cell_polarity.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Implements model with additional features defined in 
 * Nielsen, Bjarke Frost, et al. "Model to link cell shape and polarity with organogenesis." Iscience 23.2 (2020): 100830.
 * @date 2021-09-20
 * 
 */
#include "cell_polarity.h"

#include <MxUniverse.h>
#include <metrics.h>
#include <MxUtil.h>
#include <types/mx_types.h>
#include <MxLogger.h>
#include <mx_error.h>
#include <event/MxTimeEvent.h>
#include <MxBind.hpp>
#include <io/MxFIO.h>

#include <unordered_set>
#include <utility>

static int polarityVecsIdxOld = 0;
static int polarityVecsIdxCurrent = 1;

static std::string _ABColor = "blue";
static std::string _PCPColor = "green";
static float _polarityVectorScale = 0.5;
static float _polarityVectorLength = 0.5;
static bool _drawingPolarityVecs = true;

struct MxCellPolarityFIOModule : MxFIOModule {

    std::string moduleName() { return "CenterCellPolarity"; }

    HRESULT toFile(const MxMetaData &metaData, MxIOElement *fileElement);

    HRESULT fromFile(const MxMetaData &metaData, const MxIOElement &fileElement);
};

struct PolarityModelParams {
    std::string initMode = "value";
    MxVector3f initPolarAB = MxVector3f(0.0);
    MxVector3f initPolarPCP = MxVector3f(0.0);
};

struct PolarityVecsPack {
    MxVector3f v[6];

    MxVector3f &operator[](const unsigned int &idx) { return v[idx]; }
};

struct PolarityArrowsPack {
    int i[2];

    int &operator[](const unsigned int &idx) { return i[idx]; }
};

struct ParticlePolarityPack {
    PolarityVecsPack v;
    PolarityArrowsPack i;
    int32_t pId;
    bool showing;
    MxPolarityArrowData *arrowAB, *arrowPCP;

    ParticlePolarityPack() :
        showing{false}
    {}
    ParticlePolarityPack(PolarityVecsPack _v, const int32_t &_pId, const bool &_showing=true) : 
        v{_v}, pId{_pId}, arrowAB{NULL}, arrowPCP{NULL}, showing{false}
    {
        if(_showing) this->addArrows();
    }

    MxVector3f &vectorAB(const bool &current=true) {
        int idx = current ? polarityVecsIdxCurrent : polarityVecsIdxOld;
        
        return this->v[idx];
    }

    MxVector3f &vectorPCP(const bool &current=true) {
        int idx = current ? polarityVecsIdxCurrent : polarityVecsIdxOld;
        
        return this->v[idx + 2];
    }

    void cacheVectorIncrements(const MxVector3f &vecAB, const MxVector3f &vecPCP) {
        this->v[4] += vecAB;
        this->v[5] += vecPCP;
    }

    void applyVectorIncrements() {
        this->v[polarityVecsIdxOld    ] = (this->v[polarityVecsIdxCurrent    ] + this->v[4]).normalized();
        this->v[polarityVecsIdxOld + 2] = (this->v[polarityVecsIdxCurrent + 2] + this->v[5]).normalized();
        this->v[4] = MxVector3f(0.0);
        this->v[5] = MxVector3f(0.0);
    }

    void addArrows() {
        if(this->showing) return;

        auto *renderer = MxArrowRenderer::get();
        this->arrowAB = new MxPolarityArrowData();
        this->arrowPCP = new MxPolarityArrowData();
        
        this->arrowAB->scale = _polarityVectorScale;
        this->arrowAB->arrowLength = _polarityVectorLength;
        this->arrowAB->style.setColor(_ABColor);
        this->arrowPCP->scale = _polarityVectorScale;
        this->arrowPCP->arrowLength = _polarityVectorLength;
        this->arrowPCP->style.setColor(_PCPColor);
        auto idAB = renderer->addArrow(this->arrowAB);
        auto idPCP = renderer->addArrow(this->arrowPCP);

        this->i = {idAB, idPCP};
        this->showing = true;
    }

    void removeArrows() {
        if(!this->showing) return;

        auto *renderer = MxArrowRenderer::get();
        renderer->removeArrow(i[0]);
        renderer->removeArrow(i[1]);

        delete this->arrowAB;
        delete this->arrowPCP;
        this->arrowAB = NULL;
        this->arrowPCP = NULL;

        this->showing = false;
    }

    void updateArrows(const bool &current=true) {
        MxParticleHandle *ph = MxParticle_FromId(pId)->py_particle();
        MxVector3f position = ph->getPosition();

        // Scaling such that each vector appears oustide the particle with the prescribed scale
        float scaleAB = (ph->getRadius() + this->arrowAB->arrowLength) / this->arrowAB->scale;
        float scalePCP = (ph->getRadius() + this->arrowPCP->arrowLength) / this->arrowPCP->scale;

        this->arrowAB->position = position;
        this->arrowPCP->position = position;
        this->arrowAB->components = this->vectorAB(current) * scaleAB;
        this->arrowPCP->components = this->vectorPCP(current) * scalePCP;
    }
};

typedef std::unordered_map<int32_t, PolarityModelParams*> PolarityParamsType;
typedef std::vector<ParticlePolarityPack*> PartPolPackType;

int nr_partPolPack, size_partPolPack, inc_partPolPack=100;
static PartPolPackType *_partPolPack = NULL;
static PolarityParamsType *_polarityParams = NULL;

void polarityVecsFlip() {
    if(polarityVecsIdxOld == 0) {
        polarityVecsIdxOld = 1;
        polarityVecsIdxCurrent = 0;
    }
    else {
        polarityVecsIdxOld = 0;
        polarityVecsIdxCurrent = 1;
    }
}

void initPartPolPack() {
    if(_partPolPack) return;
    _partPolPack = new PartPolPackType();
    _partPolPack->resize(inc_partPolPack, NULL);
    size_partPolPack = inc_partPolPack;
    nr_partPolPack = 0;
}

ParticlePolarityPack *insertPartPolPack(const int &pId, const PolarityVecsPack &pvp) {
    while (pId >= size_partPolPack) {
        size_partPolPack += inc_partPolPack;
        _partPolPack->resize(size_partPolPack, NULL);
    }

    if(pId < nr_partPolPack && (*_partPolPack)[pId] != NULL) {
        mx_exp(std::invalid_argument("polarity parameters already set!"));
        return NULL;
    }

    auto ppp = new ParticlePolarityPack(pvp, pId);
    if(ppp->showing) ppp->updateArrows();
    (*_partPolPack)[pId] = ppp;
    nr_partPolPack++;
    return ppp;
}

void removePartPolPack(const int &pId) {
    auto p = (*_partPolPack)[pId];

    if(!p) {
        mx_exp(std::invalid_argument("polarity parameters not set!"));
        return;
    }

    if(p->showing) p->removeArrows();
    delete p;
    (*_partPolPack)[pId] = NULL;
    nr_partPolPack--;
}

std::pair<MxVector3f, MxVector3f> initPolarityVec(const int &pId) {
    MxParticle *p = MxParticle_FromId(pId);
    if(!p) return std::make_pair(MxVector3f(0.0), MxVector3f(0.0));
    
    auto itrPolarityParams = _polarityParams->find(p->typeId);
    if(itrPolarityParams == _polarityParams->end()) {
        Log(LOG_TRACE) << "No known particle type for initializing polar particle: " << pId << ", " << p->typeId;

        return std::make_pair(MxVector3f(0.0), MxVector3f(0.0));
    }

    PolarityModelParams *pmp = itrPolarityParams->second;
    MxVector3f ivAB, ivPCP;
    if(strcmp(pmp->initMode.c_str(), "value") == 0) {
        ivAB = MxVector3f(pmp->initPolarAB);
        ivPCP = MxVector3f(pmp->initPolarPCP);
    }
    else if(strcmp(pmp->initMode.c_str(), "random") == 0) {
        ivAB = MxRandomPoint(MxPointsType::Sphere);
        ivPCP = MxRandomPoint(MxPointsType::Sphere);
    }
    Log(LOG_TRACE) << "Initialized particle " << pId << ": " << ivAB << ", " << ivPCP;

    insertPartPolPack(pId, {ivAB, ivAB, ivPCP, ivPCP, MxVector3f(0.0), MxVector3f(0.0)});

    return std::make_pair(ivAB, ivPCP);
}

void MxCellPolarity_register(const int &pId) {
    if(pId >= nr_partPolPack || (*_partPolPack)[pId] == NULL) initPolarityVec(pId);
}

void MxCellPolarity_register(MxParticleHandle *ph) {
    if(!ph) return;

    MxCellPolarity_register(ph->id);
}

void MxCellPolarity_unregister(MxParticleHandle *ph) {
    removePartPolPack(ph->id);
}

void MxCellPolarity_registerType(MxParticleType *pType, 
                                 const std::string &initMode, 
                                 const MxVector3f &initPolarAB, 
                                 const MxVector3f &initPolarPCP) {
    if(!pType) return;

    auto itr = _polarityParams->find(pType->id);
    if(itr != _polarityParams->end()) {
        _polarityParams->erase(itr);
        delete itr->second;
    }

    PolarityModelParams *pmp = new PolarityModelParams();
    pmp->initMode = initMode;
    pmp->initPolarAB = initPolarAB;
    pmp->initPolarPCP = initPolarPCP;
    (*_polarityParams)[pType->id] = pmp;
}

MxVector3f MxCellPolarity_GetVectorAB(const int &pId, const bool &current) {
    return (*_partPolPack)[pId]->vectorAB(current);
}

void MxCellPolarity_SetVectorAB(const int &pId, const MxVector3f &pVec, const bool &current, const bool &init) {
    auto pp = (*_partPolPack)[pId];
    pp->v[current ? polarityVecsIdxCurrent : polarityVecsIdxOld] = pVec;
    pp->updateArrows(current);
    if (init) MxCellPolarity_SetVectorAB(pId, pVec, !current, false);
}

MxVector3f MxCellPolarity_GetVectorPCP(const int &pId, const bool &current) {
    return (*_partPolPack)[pId]->vectorPCP(current);
}

void MxCellPolarity_SetVectorPCP(const int &pId, const MxVector3f &pVec, const bool &current, const bool &init) {
    auto pp = (*_partPolPack)[pId];
    auto idx = current ? polarityVecsIdxCurrent : polarityVecsIdxOld;
    pp->v[idx + 2] = pVec;
    pp->updateArrows(current);
    if (init) MxCellPolarity_SetVectorPCP(pId, pVec, !current, false);
}

void cacheVectorIncrements(const int &pId, const MxVector3f &vecAB, const MxVector3f &vecPCP) {
    (*_partPolPack)[pId]->cacheVectorIncrements(vecAB, vecPCP);
}

void applyVectorIncrements(const int &pId) {
    (*_partPolPack)[pId]->applyVectorIncrements();
}

const std::string MxCellPolarity_GetInitMode(MxParticleType *pType) {
    if(!pType) return "";

    auto itr = _polarityParams->find(pType->id);
    if(itr == _polarityParams->end()) return "";

    return itr->second->initMode;
}

void MxCellPolarity_SetInitMode(MxParticleType *pType, const std::string &value) {
    if(!pType) return;

    auto itr = _polarityParams->find(pType->id);
    if(itr == _polarityParams->end()) return;

    itr->second->initMode = value;
}

const MxVector3f MxCellPolarity_GetInitPolarAB(MxParticleType *pType) {
    if(!pType) return MxVector3f(0.0);

    auto itr = _polarityParams->find(pType->id);
    if(itr == _polarityParams->end()) return MxVector3f(0.0);

    return itr->second->initPolarAB;
}

void MxCellPolarity_SetInitPolarAB(MxParticleType *pType, const MxVector3f &value) {
    if(!pType) return;

    auto itr = _polarityParams->find(pType->id);
    if(itr == _polarityParams->end()) return;

    itr->second->initPolarAB = value;
}

const MxVector3f MxCellPolarity_GetInitPolarPCP(MxParticleType *pType) {
    if(!pType) return MxVector3f(0.0);

    auto itr = _polarityParams->find(pType->id);
    if(itr == _polarityParams->end()) return MxVector3f(0.0);

    return itr->second->initPolarPCP;
}

void MxCellPolarity_SetInitPolarPCP(MxParticleType *pType, const MxVector3f &value) {
    if(!pType) return;

    auto itr = _polarityParams->find(pType->id);
    if(itr == _polarityParams->end()) return;

    itr->second->initPolarPCP = value;
}

void eval_polarity_force_persistent(struct MxForce *force, struct MxParticle *p, int stateVectorId, FPTYPE *f) {
    PolarityForcePersistent *pf = (PolarityForcePersistent*)force;

    auto ppp = (*_partPolPack)[p->id];
    MxVector3f polAB = ppp->vectorAB();
    MxVector3f polPCP = ppp->vectorPCP();

    for(int i = 0; i < 3; i++) f[i] += pf->sensAB * polAB[i] + pf->sensPCP * polPCP[i];
}


static std::vector<PolarityForcePersistent*> *storedPersistentForces = NULL;

static void storePersistentForce(PolarityForcePersistent *f) {
    if(f == NULL) 
        return;

    if(storedPersistentForces == NULL) 
        storedPersistentForces = new std::vector<PolarityForcePersistent*>();

    for(auto sf : *storedPersistentForces) 
        if(sf == f) 
            return;

    storedPersistentForces->push_back(f);

}

static void unstorePersistentForce(PolarityForcePersistent *f) {
    if(f == NULL || storedPersistentForces == NULL) 
        return;

    auto itr = std::find(storedPersistentForces->begin(), storedPersistentForces->end(), f);
    if(itr != storedPersistentForces->end()) 
        storedPersistentForces->erase(itr);
}

PolarityForcePersistent::~PolarityForcePersistent() {
    unstorePersistentForce(this);
}

PolarityForcePersistent *MxCellPolarity_createForce_persistent(const float &sensAB, const float &sensPCP) {
    PolarityForcePersistent *pf = new PolarityForcePersistent();

    storePersistentForce(pf);

    pf->func = (MxForce_OneBodyPtr)eval_polarity_force_persistent;
    pf->sensAB = sensAB;
    pf->sensPCP = sensPCP;
    return pf;
}

static inline MxMatrix3f tensorProduct(const MxVector3f &rowVec, const MxVector3f &colVec) {
    MxMatrix3f result(1.0);
    for(int j = 0; j < 3; ++j)
        for(int i = 0; i < 3; ++i)
            result[j][i] *= rowVec[i] * colVec[j];
    return result;
}

void MxCellPolarity_SetDrawVectors(const bool &_draw) {
    _drawingPolarityVecs = _draw;

    if (_draw) for(auto p : *_partPolPack) p->addArrows();
    else for(auto p : *_partPolPack) p->removeArrows();
}

void MxCellPolarity_SetArrowColors(const std::string &colorAB, const std::string &colorPCP) {
    _ABColor = colorAB;
    _PCPColor = colorPCP;

    for(auto p : *_partPolPack) {
        if(!p) continue;

        p->arrowAB->style.setColor(_ABColor);
        p->arrowPCP->style.setColor(_PCPColor);
    }
}

void MxCellPolarity_SetArrowScale(const float &_scale) {
    _polarityVectorScale = _scale;

    for(auto p : *_partPolPack) {
        if(!p) continue;

        p->arrowAB->scale = _polarityVectorScale;
        p->arrowPCP->scale = _polarityVectorScale;
    }
}

void MxCellPolarity_SetArrowLength(const float &_length) {
    _polarityVectorLength = _length;

    for(auto p : *_partPolPack) {
        if(!p) continue;

        p->arrowAB->arrowLength = _length;
        p->arrowPCP->arrowLength = _length;
    }
}

MxPolarityArrowData *MxCellPolarity_GetVectorArrowAB(const int32_t &pId) {
    return (*_partPolPack)[pId]->arrowAB;
}

MxPolarityArrowData *MxCellPolarity_GetVectorArrowPCP(const int32_t &pId) {
    return (*_partPolPack)[pId]->arrowPCP;
}

void updatePolariyVectorArrows(const int32_t &pId) {
    (*_partPolPack)[pId]->updateArrows();
}

void removePolarityVectorArrows(const int &pId) {
    Log(LOG_DEBUG) << "";
    
    (*_partPolPack)[pId]->removeArrows();
}

HRESULT MxCellPolarity_run(const MxTimeEvent &event) {
    MxCellPolarity_update();
    return S_OK;
}

static bool _loaded = false;

void MxCellPolarity_load() {
    if(_loaded) return;

    // Instantiate i/o module and register for export
    MxCellPolarityFIOModule *ioModule = new MxCellPolarityFIOModule();
    ioModule->registerIOModule();

    // initialize all module variables
    _polarityParams = new PolarityParamsType();
    initPartPolPack();

    // import data if available
    if(MxFIO::hasImport()) ioModule->load();
    
    // load callback to execute model along with simulation
    MxTimeEventMethod *fcn = new MxTimeEventMethod(MxCellPolarity_run);
    MxOnTimeEvent(getUniverse()->getDt(), fcn);

    _loaded = true;
}

void MxCellPolarity_update() {
    int i;
    ParticlePolarityPack *p;
#pragma omp parallel for schedule(static), private(i,p,size_partPolPack,_partPolPack,_drawingPolarityVecs)
    for(i = 0; i < size_partPolPack; i++) {
        p = (*_partPolPack)[i];
        if(!p) continue;

        p->applyVectorIncrements();
        if(_drawingPolarityVecs) p->updateArrows();
    }

    polarityVecsFlip();

    Log(LOG_DEBUG) << "";
}

void eval_potential_cellpolarity(struct MxPotential *p, 
                                 struct MxParticle *part_i, 
                                 struct MxParticle *part_j, 
                                 FPTYPE *dx, 
                                 FPTYPE r2, 
                                 FPTYPE *e, 
                                 FPTYPE *f) {
    MxCellPolarityPotentialContact *pot = (MxCellPolarityPotentialContact*)p;
    auto ppi = (*_partPolPack)[part_i->id];
    auto ppj = (*_partPolPack)[part_j->id];

    if(r2 > pot->b * pot->b) return;

    MxVector3f rel_pos = - MxVector3f::from(dx);
    float len_r = std::sqrt(r2);
    MxVector3f rh = rel_pos / len_r;

    MxVector3f pi = ppi->v[polarityVecsIdxCurrent];
    MxVector3f qi = ppi->v[polarityVecsIdxCurrent + 2];
    MxVector3f pj = ppj->v[polarityVecsIdxCurrent];
    MxVector3f qj = ppj->v[polarityVecsIdxCurrent + 2];

    float g = 0.0;
    MxVector3f dgdrh(0.0), dgdpi(0.0), dgdqi(0.0), dgdpj(0.0), dgdqj(0.0);
    MxVector3f v1, v2;

    if(pot->couplingFlat > 0.0) {
        float len_v1, len_v2, u1, u2, u3, u4;
        MxVector3f pti(0.0), ptj(0.0);
        MxVector3f v3;
        MxMatrix3f dptidpi(0.0), dptjdpj(0.0);
        MxMatrix3f eye = Magnum::Matrix3x3{Magnum::Math::IdentityInit};
        MxMatrix3f ir = eye - tensorProduct(rh, rh);
        MxMatrix3f ptiptj;

        switch(pot->cType) {
            case PolarContactType::REGULAR : {
                pti = pi;
                ptj = pj;
                
                u1 = rh.dot(pti);
                u2 = rh.dot(ptj);
                dgdrh += pot->couplingFlat * (2.0 * pti.dot(ptj) * rh - u2 * pti - u1 * ptj);
                dgdpi += pot->couplingFlat * (ptj - u2 * rh);
                dgdpj += pot->couplingFlat * (pti - u1 * rh);
                break;
            }
            case PolarContactType::ISOTROPIC : {
                v3 = - pot->bendingCoeff * rh;
                v1 = pi - v3;
                len_v1 = v1.length();
                
                if(len_v1 > 0.0) {

                    v2 = pj + v3;
                    len_v2 = v2.length();

                    if(len_v2 > 0.0) {
                        pti = v1 / len_v1;
                        ptj = v2 / len_v2;

                        u1 = rh.dot(pti);
                        u2 = rh.dot(ptj);
                        u3 = pti.dot(ptj);
                        u4 = u1 * u2 - u3;
                        dgdrh += pot->couplingFlat * (2.0 * u3 * rh - u2 * pti - u1 * ptj);
                        dgdpi += pot->couplingFlat * (ptj - u2 * rh + u4 * pti) / len_v1;
                        dgdpj += pot->couplingFlat * (pti - u1 * rh + u4 * ptj) / len_v2;
                    }
                }

                break;
            }
            case PolarContactType::ANISOTROPIC : {
                v3 = - 0.5 * (qi + qj);
                v3 *= pot->bendingCoeff * rh.dot(v3);

                v1 = pi + v3;
                len_v1 = v1.length();
                
                if(len_v1 > 0.0) {
                    
                    v2 = pj - v3;
                    len_v2 = v2.length();

                    if(len_v2 > 0.0) {
                        pti = v1 / len_v1;
                        ptj = v2 / len_v2;

                        dptidpi = (eye - tensorProduct(pti, pti)) / len_v1;
                        dptjdpj = (eye - tensorProduct(ptj, ptj)) / len_v2;

                        ptiptj = tensorProduct(pti, ptj);
                        ptiptj = ptiptj + ptiptj.transposed();
                        dgdrh += 2.0 * pot->couplingFlat * pti.dot(ptj) * rh - ptiptj * rh;
                        dgdpi += pot->couplingFlat * (dptidpi * ir * ptj);
                        dgdpj += pot->couplingFlat * (dptjdpj * ir * pti);
                    }
                }
                
                break;
            }
        }

        v1 = Magnum::Math::cross(rh, pti);
        v2 = Magnum::Math::cross(rh, ptj);
        g += pot->couplingFlat * v1.dot(v2);
    }

    if(pot->couplingOrtho > 0.0) {
        v1 = Magnum::Math::cross(pi, qi);
        v2 = Magnum::Math::cross(pj, qj);

        g += v1.dot(v2);

        float pipj = pi.dot(pj);
        float qiqj = qi.dot(qj);
        float piqj = pi.dot(qj);
        float pjqi = pj.dot(qi);
        dgdpi += pot->couplingOrtho * (qiqj * pj - pjqi * qj);
        dgdqi += pot->couplingOrtho * (pipj * qj - piqj * pj);
        dgdpj += pot->couplingOrtho * (qiqj * pi - piqj * qi);
        dgdqj += pot->couplingOrtho * (pipj * qi - pjqi * pi);
    }

    if(pot->couplingLateral > 0.0) {
        v1 = Magnum::Math::cross(rh, qi);
        v2 = Magnum::Math::cross(rh, qj);

        g += pot->couplingLateral * (v1.dot(v2));

        float rhqi = rh.dot(qi);
        float rhqj = rh.dot(qj);
        dgdrh += pot->couplingLateral * (2.0 * qi.dot(qj) * rh - rhqi * qj - rhqj * qi);
        dgdqi += pot->couplingLateral * (qj - rhqj * rh);
        dgdqj += pot->couplingLateral * (qi - rhqi * rh);
    }

    float powTerm = std::pow(M_E, -len_r / pot->distanceCoeff);
    MxVector3f dgdr = (rh.dot(dgdrh) * rh - dgdrh) / len_r;

    MxVector3f incForce = pot->mag * powTerm * (g / pot->distanceCoeff * rh + dgdr);

    // No flipping here. Needs to be done in update function

    float polmag = powTerm * pot->rate * getUniverse()->getDt();
    
    ppi->cacheVectorIncrements(polmag * dgdpi, polmag * dgdqi);
    ppj->cacheVectorIncrements(polmag * dgdpj, polmag * dgdqj);

    *e += powTerm * g;
    f[0] += incForce[0];
    f[1] += incForce[1];
    f[2] += incForce[2];

}

MxCellPolarityPotentialContact::MxCellPolarityPotentialContact() : 
    MxPotential()
{
    this->kind = POTENTIAL_KIND_BYPARTICLES;
    this->eval_byparts = MxPotentialEval_ByParticles(eval_potential_cellpolarity);
}

static std::unordered_map<std::string, PolarContactType> polarContactTypeMap {
    {"regular", PolarContactType::REGULAR}, 
    {"isotropic", PolarContactType::ISOTROPIC}, 
    {"anisotropic", PolarContactType::ANISOTROPIC}
};

static std::vector<MxCellPolarityPotentialContact*> *storedContactPotentials = NULL;

static void storeContactPotential(MxCellPolarityPotentialContact *p) {
    if(p == NULL) 
        return;

    if(storedContactPotentials == NULL) 
        storedContactPotentials = new std::vector<MxCellPolarityPotentialContact*>();

    for(auto sp : *storedContactPotentials) 
        if(sp == p) 
            return;

    storedContactPotentials->push_back(p);

}

static void unstoreContactPotential(MxCellPolarityPotentialContact *p) {
    if(p == NULL || storedContactPotentials == NULL) 
        return;

    auto itr = std::find(storedContactPotentials->begin(), storedContactPotentials->end(), p);
    if(itr != storedContactPotentials->end()) 
        storedContactPotentials->erase(itr);
}

MxCellPolarityPotentialContact *_boundContactPotential = NULL;

static void _boundUnstoreContactPotential(MxPotential *p) {
    MxCellPolarityPotentialContact *pp = _boundContactPotential;
    unstoreContactPotential(pp);
}

static MxPotentialClear bindUnstoreContactPotential(MxCellPolarityPotentialContact *p) {
    _boundContactPotential = p;
    return MxPotentialClear(_boundUnstoreContactPotential);
}

MxCellPolarityPotentialContact *potential_create_cellpolarity(const float &cutoff, 
                                                              const float &mag, 
                                                              const float &rate,
                                                              const float &distanceCoeff, 
                                                              const float &couplingFlat, 
                                                              const float &couplingOrtho, 
                                                              const float &couplingLateral, 
                                                              std::string contactType, 
                                                              const float &bendingCoeff)
{
    MxCellPolarityPotentialContact *pot = new MxCellPolarityPotentialContact();

    auto tItr = polarContactTypeMap.find(contactType);
    if(tItr == polarContactTypeMap.end()) mx_exp(std::runtime_error("Invalid type"));

    storeContactPotential(pot);
    pot->clear_func = bindUnstoreContactPotential(pot);

    pot->mag = mag;
    pot->rate = rate;
    pot->distanceCoeff = distanceCoeff;
    pot->couplingFlat = couplingFlat;
    pot->couplingOrtho = couplingOrtho;
    pot->couplingLateral = couplingLateral;
    pot->cType = tItr->second;
    pot->bendingCoeff = bendingCoeff;

    pot->a = std::sqrt(std::numeric_limits<float>::epsilon());
    pot->b = cutoff;
    pot->name = "Cell Polarity Contact";

    Log(LOG_TRACE) << "";

    return pot;
}


namespace mx { namespace io {

#define MXCENTERCELLPOLARITYIOTOEASY(fe, key, member) \
    fe = new MxIOElement(); \
    if(mx::io::toFile(member, metaData, fe) != S_OK)  \
        return E_FAIL; \
    fe->parent = fileElement; \
    fileElement->children[key] = fe;

#define MXCENTERCELLPOLARITYIOFROMEASY(feItr, children, metaData, key, member_p) \
    feItr = children.find(key); \
    if(feItr == children.end() || mx::io::fromFile(*feItr->second, metaData, member_p) != S_OK) \
        return E_FAIL;

template <>
HRESULT toFile(const PolarityModelParams &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    MxIOElement *fe;

    MXCENTERCELLPOLARITYIOTOEASY(fe, "initMode", dataElement.initMode);
    MXCENTERCELLPOLARITYIOTOEASY(fe, "initPolarAB", dataElement.initPolarAB);
    MXCENTERCELLPOLARITYIOTOEASY(fe, "initPolarPCP", dataElement.initPolarPCP);

    fileElement->type = "PolarityParameters";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, PolarityModelParams *dataElement) {

    MxIOChildMap::const_iterator feItr;

    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "initMode", &dataElement->initMode);
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "initPolarAB", &dataElement->initPolarAB);
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "initPolarPCP", &dataElement->initPolarPCP);

    return S_OK;
}

template <>
HRESULT toFile(const PolarityVecsPack &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    MxIOElement *fe;

    MXCENTERCELLPOLARITYIOTOEASY(fe, "v0", dataElement.v[0]);
    MXCENTERCELLPOLARITYIOTOEASY(fe, "v1", dataElement.v[1]);
    MXCENTERCELLPOLARITYIOTOEASY(fe, "v2", dataElement.v[2]);
    MXCENTERCELLPOLARITYIOTOEASY(fe, "v3", dataElement.v[3]);
    MXCENTERCELLPOLARITYIOTOEASY(fe, "v4", dataElement.v[4]);
    MXCENTERCELLPOLARITYIOTOEASY(fe, "v5", dataElement.v[5]);

    fileElement->type = "PolarityVecs";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, PolarityVecsPack *dataElement) {

    MxIOChildMap::const_iterator feItr;

    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "v0", &dataElement->v[0]);
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "v1", &dataElement->v[1]);
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "v2", &dataElement->v[2]);
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "v3", &dataElement->v[3]);
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "v4", &dataElement->v[4]);
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "v5", &dataElement->v[5]);

    return S_OK;
}

template <>
HRESULT toFile(const ParticlePolarityPack &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    MxIOElement *fe;

    MXCENTERCELLPOLARITYIOTOEASY(fe, "polarityVectors", dataElement.v);
    MXCENTERCELLPOLARITYIOTOEASY(fe, "particleId", dataElement.pId);
    MXCENTERCELLPOLARITYIOTOEASY(fe, "showing", dataElement.showing);

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, ParticlePolarityPack *dataElement) {

    MxIOChildMap::const_iterator feItr;

    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "polarityVectors", &dataElement->v);
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "particleId", &dataElement->pId);
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "showing", &dataElement->showing);

    return S_OK;
}

template <>
HRESULT toFile(const PolarityForcePersistent &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    MxIOElement *fe;

    MXCENTERCELLPOLARITYIOTOEASY(fileElement, "sensAB", dataElement.sensAB);
    MXCENTERCELLPOLARITYIOTOEASY(fileElement, "sensPCP", dataElement.sensPCP);

    fileElement->type = "PersistentForce";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, PolarityForcePersistent **dataElement) {

    MxIOChildMap::const_iterator feItr;

    float sensAB, sensPCP;

    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "sensAB", &sensAB);
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "sensPCP", &sensPCP);

    *dataElement = MxCellPolarity_createForce_persistent(sensAB, sensPCP);

    return S_OK;
}

template <>
HRESULT toFile(const MxCellPolarityPotentialContact &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    MxIOElement *fe;

    MXCENTERCELLPOLARITYIOTOEASY(fe, "cutoff", dataElement.b);
    MXCENTERCELLPOLARITYIOTOEASY(fe, "couplingFlat", dataElement.couplingFlat);
    MXCENTERCELLPOLARITYIOTOEASY(fe, "couplingOrtho", dataElement.couplingOrtho);
    MXCENTERCELLPOLARITYIOTOEASY(fe, "couplingLateral", dataElement.couplingLateral);
    MXCENTERCELLPOLARITYIOTOEASY(fe, "distanceCoeff", dataElement.distanceCoeff);
    MXCENTERCELLPOLARITYIOTOEASY(fe, "cType", (unsigned int)dataElement.cType);
    MXCENTERCELLPOLARITYIOTOEASY(fe, "mag", dataElement.mag);
    MXCENTERCELLPOLARITYIOTOEASY(fe, "rate", dataElement.rate);
    MXCENTERCELLPOLARITYIOTOEASY(fe, "bendingCoeff", dataElement.bendingCoeff);

    fileElement->type = "ContactPotential";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxCellPolarityPotentialContact **dataElement) {

    MxIOChildMap::const_iterator feItr;

    float cutoff, couplingFlat, couplingOrtho, couplingLateral, distanceCoeff, mag, rate, bendingCoeff;
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "cutoff", &cutoff);
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "couplingFlat", &couplingFlat);
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "couplingOrtho", &couplingOrtho);
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "couplingLateral", &couplingLateral);
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "distanceCoeff", &distanceCoeff);
    
    unsigned int cTypeUI;
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "cType", &cTypeUI);
    PolarContactType cType = (PolarContactType)cTypeUI;
    std::string cTypeName = "regular";
    for(auto &cTypePair : polarContactTypeMap) 
        if(cTypePair.second == cType) 
            cTypeName = cTypePair.first;

    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "mag", &mag);
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "rate", &rate);
    MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "bendingCoeff", &bendingCoeff);

    *dataElement = potential_create_cellpolarity(cutoff, mag, rate, distanceCoeff, couplingFlat, couplingOrtho, couplingLateral, cTypeName, bendingCoeff);

    return S_OK;
}

}};

static bool recursivePotentialCompare(MxCellPolarityPotentialContact *pp, MxPotential *p) {
    if(!pp || !p) 
        return false;

    if(p->kind == POTENTIAL_KIND_COMBINATION && p->flags & POTENTIAL_SUM) {
        if(recursivePotentialCompare(pp, p->pca)) return true;

        if(recursivePotentialCompare(pp, p->pcb)) return true;

        return false;
    } 
    else
        return pp == p;

}

static bool recursiveForceCompare(PolarityForcePersistent *pf, MxForce *f) {
    if(!pf || !f) 
        return false;

    if(f->type == FORCE_SUM) {
        MxForceSum *sf = (MxForceSum*)f;
        if(recursiveForceCompare(pf, sf->f1)) return true;

        if(recursiveForceCompare(pf, sf->f2)) return true;

        return false;
    } 
    else return pf == f;
}

HRESULT MxCellPolarityFIOModule::toFile(const MxMetaData &metaData, MxIOElement *fileElement) {

    MxIOElement *fe;

    // Store registered types

    if(_polarityParams != NULL) {
        std::vector<PolarityModelParams> polarityParams;
        std::vector<int32_t> polarTypes;
        for(auto &typePair : *_polarityParams) {
            if(typePair.second != NULL) {
                polarTypes.push_back(typePair.first);
                polarityParams.push_back(*typePair.second);
            }
        }

        if(polarityParams.size() > 0) {
            MXCENTERCELLPOLARITYIOTOEASY(fe, "polarTypes", polarTypes);
            MXCENTERCELLPOLARITYIOTOEASY(fe, "polarityParams", polarityParams);
        }
    }

    // Store potentials
    
    if(storedContactPotentials != NULL && storedContactPotentials->size() > 0) {

        std::vector<MxCellPolarityPotentialContact> potentials;
        auto numPots = storedContactPotentials->size();
        std::vector<std::vector<uint16_t> > potentialTypesA(numPots, std::vector<uint16_t>()), potentialTypesB(numPots, std::vector<uint16_t>());
        std::vector<std::vector<uint16_t> > potentialTypesClusterA(numPots, std::vector<uint16_t>()), potentialTypesClusterB(numPots, std::vector<uint16_t>());

        for(unsigned int pi = 0; pi < numPots; pi++) 
            potentials.push_back(*(*storedContactPotentials)[pi]);

        for(unsigned int i = 0; i < _Engine.nr_types; i++) {
            for(unsigned int j = i; j < _Engine.nr_types; j++) {
                unsigned int k = i * _Engine.max_type + j;

                MxPotential *p = _Engine.p[k], *pc = _Engine.p_cluster[k];

                for(unsigned int pi = 0; pi < numPots; pi++) {
                    MxCellPolarityPotentialContact *pp = (*storedContactPotentials)[pi];
                    if(recursivePotentialCompare(pp, p)) {
                        potentialTypesA[pi].push_back(i);
                        potentialTypesB[pi].push_back(j);
                    }
                    if(recursivePotentialCompare(pp, pc)) {
                        potentialTypesClusterA[pi].push_back(i);
                        potentialTypesClusterB[pi].push_back(j);
                    }
                }

            }
        }

        MXCENTERCELLPOLARITYIOTOEASY(fe, "potentials", potentials);
        MXCENTERCELLPOLARITYIOTOEASY(fe, "potentialTypesA", potentialTypesA);
        MXCENTERCELLPOLARITYIOTOEASY(fe, "potentialTypesB", potentialTypesB);
        MXCENTERCELLPOLARITYIOTOEASY(fe, "potentialTypesClusterA", potentialTypesClusterA);
        MXCENTERCELLPOLARITYIOTOEASY(fe, "potentialTypesClusterB", potentialTypesClusterB);

    }

    // Store forces

    if(storedPersistentForces != NULL && storedPersistentForces->size() > 0) {
        std::vector<PolarityForcePersistent> forces;
        std::vector<int16_t> forceTypes(storedPersistentForces->size(), -1);

        for(unsigned int i = 0; i < storedPersistentForces->size(); i++) 
            forces.push_back(*(*storedPersistentForces)[i]);

        for(unsigned int i = 0; i < _Engine.nr_types; i++) 
            for(unsigned int fi = 0; fi < storedPersistentForces->size(); fi++) 
                if(recursiveForceCompare((*storedPersistentForces)[fi], _Engine.p_singlebody->force)) 
                    forceTypes[fi] = i;

        MXCENTERCELLPOLARITYIOTOEASY(fe, "forces", forces);
        MXCENTERCELLPOLARITYIOTOEASY(fe, "forceTypes", forceTypes);
    }
    
    // Store states of registered particles

    if(_partPolPack != NULL && _partPolPack->size() > 0) {
        std::vector<ParticlePolarityPack> particles;
        for(auto &pp : *_partPolPack) 
            if(pp) 
                particles.push_back(*pp);

        MXCENTERCELLPOLARITYIOTOEASY(fe, "particles", particles);
    }

    return S_OK;
}

HRESULT MxCellPolarityFIOModule::fromFile(const MxMetaData &metaData, const MxIOElement &fileElement) {

    MxIOChildMap::const_iterator feItr;

    // Load registered types

    if(fileElement.children.find("polarityParams") != fileElement.children.end()) {
    
        std::vector<PolarityModelParams> polarityParams;
        std::vector<int32_t> polarTypes;

        MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "polarityParams", &polarityParams);
        MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "polarTypes", &polarTypes);

        for(unsigned int i = 0; i < polarTypes.size(); i++) {
            unsigned int pTypeId = MxFIO::importSummary->particleTypeIdMap[polarTypes[i]];
            auto pmp = polarityParams[i];

            MxParticleType *pType = &_Engine.types[pTypeId];
            if(pType == NULL) {
                mx_exp(std::runtime_error("Particle type not defined"));
                return E_FAIL;
            }
            MxCellPolarity_registerType(pType, pmp.initMode, pmp.initPolarAB, pmp.initPolarPCP);
        }

    }

    // Load potentials

    if(fileElement.children.find("potentials") != fileElement.children.end()) {

        std::vector<MxCellPolarityPotentialContact*> potentials;
        std::vector<std::vector<unsigned int> > potentialTypesA, potentialTypesB, potentialTypesClusterA, potentialTypesClusterB;

        MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "potentials", &potentials);
        MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "potentialTypesA", &potentialTypesA);
        MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "potentialTypesB", &potentialTypesB);
        MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "potentialTypesClusterA", &potentialTypesClusterA);
        MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "potentialTypesClusterB", &potentialTypesClusterB);

        for(unsigned int i = 0; i < potentials.size(); i++) {
            auto p = potentials[i];
            std::vector<unsigned int> pTypesIdA = potentialTypesA[i];
            std::vector<unsigned int> pTypesIdB = potentialTypesB[i];
            std::vector<unsigned int> pTypesIdClusterA = potentialTypesClusterA[i];
            std::vector<unsigned int> pTypesIdClusterB = potentialTypesClusterB[i];

            for(unsigned int j = 0; j < pTypesIdA.size(); j++) { 
                MxParticleType *pTypeA = &_Engine.types[MxFIO::importSummary->particleTypeIdMap[pTypesIdA[j]]];
                MxParticleType *pTypeB = &_Engine.types[MxFIO::importSummary->particleTypeIdMap[pTypesIdB[j]]];
                if(!pTypeA || !pTypeB) {
                    mx_exp(std::runtime_error("Particle type not defined"));
                    return E_FAIL;
                }
                MxBind::types(p, pTypeA, pTypeB);
            }

            for(unsigned int j = 0; j < pTypesIdClusterA.size(); j++) { 
                MxParticleType *pTypeA = &_Engine.types[MxFIO::importSummary->particleTypeIdMap[pTypesIdClusterA[j]]];
                MxParticleType *pTypeB = &_Engine.types[MxFIO::importSummary->particleTypeIdMap[pTypesIdClusterB[j]]];
                if(!pTypeA || !pTypeB) {
                    mx_exp(std::runtime_error("Particle type not defined"));
                    return E_FAIL;
                }
                MxBind::types(p, pTypeA, pTypeB, true);
            }
        }

    }

    // Load forces

    if(fileElement.children.find("forces") != fileElement.children.end()) {

        std::vector<PolarityForcePersistent*> forces;
        std::vector<int16_t> forceTypes;

        MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "forces", &forces);
        MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "forceTypes", &forceTypes);

        for(unsigned int i = 0; i < forces.size(); i++) {
            auto pTypeIdImport = forceTypes[i];
            if(pTypeIdImport < 0) 
                continue;

            auto pTypeId = MxFIO::importSummary->particleTypeIdMap[pTypeIdImport];

            MxParticleType *pType = &_Engine.types[pTypeId];
            if(pType == NULL) {
                mx_exp(std::runtime_error("Particle type not defined"));
                return E_FAIL;
            }

            MxBind::force(forces[i], pType);
        }

    }
    
    // Load states of registered particles

    if(fileElement.children.find("particles") != fileElement.children.end()) {

        std::vector<ParticlePolarityPack> particles;

        MXCENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "particles", &particles);

        for(unsigned int i = 0; i < particles.size(); i++) {
            auto ppp = particles[i];
            auto pId = MxFIO::importSummary->particleIdMap[ppp.pId];
            MxCellPolarity_register(pId);
            auto pppe = (*_partPolPack)[pId];
            if(ppp.showing) 
                pppe->addArrows();

        }

    }

    return S_OK;
}


using namespace mx::models::center;

MxVector3f CellPolarity::getVectorAB(const int &pId, const bool &current) {
    return MxCellPolarity_GetVectorAB(pId, current);
}

MxVector3f CellPolarity::getVectorPCP(const int &pId, const bool &current) {
    return MxCellPolarity_GetVectorPCP(pId, current);
}

void CellPolarity::setVectorAB(const int &pId, const MxVector3f &pVec, const bool &current, const bool &init) {
    return MxCellPolarity_SetVectorAB(pId, pVec, current, init);
}

void CellPolarity::setVectorPCP(const int &pId, const MxVector3f &pVec, const bool &current, const bool &init) {
    return MxCellPolarity_SetVectorPCP(pId, pVec, current, init);
}

void CellPolarity::update() {
    return MxCellPolarity_update();
}

void CellPolarity::registerParticle(MxParticleHandle *ph) {
    return MxCellPolarity_register(ph);
}

void CellPolarity::unregister(MxParticleHandle *ph) {
    return MxCellPolarity_unregister(ph);
}

void CellPolarity::registerType(MxParticleType *pType, 
                                const std::string &initMode, 
                                const MxVector3f &initPolarAB, 
                                const MxVector3f &initPolarPCP) 
{
    return MxCellPolarity_registerType(pType, initMode, initPolarAB, initPolarPCP);
}

const std::string CellPolarity::getInitMode(MxParticleType *pType) {
    return MxCellPolarity_GetInitMode(pType);
}

void CellPolarity::setInitMode(MxParticleType *pType, const std::string &value) {
    return MxCellPolarity_SetInitMode(pType, value);
}

const MxVector3f CellPolarity::getInitPolarAB(MxParticleType *pType) {
    return MxCellPolarity_GetInitPolarAB(pType);
}

void CellPolarity::setInitPolarAB(MxParticleType *pType, const MxVector3f &value) {
    return MxCellPolarity_SetInitPolarAB(pType, value);
}

const MxVector3f CellPolarity::getInitPolarPCP(MxParticleType *pType) {
    return MxCellPolarity_GetInitPolarPCP(pType);
}

void CellPolarity::setInitPolarPCP(MxParticleType *pType, const MxVector3f &value) {
    return MxCellPolarity_SetInitPolarPCP(pType, value);
}

PolarityForcePersistent *CellPolarity::forcePersistent(const float &sensAB, const float &sensPCP) {
    return MxCellPolarity_createForce_persistent(sensAB, sensPCP);
}

void CellPolarity::setDrawVectors(const bool &_draw) {
    MxCellPolarity_SetDrawVectors(_draw);
}

void CellPolarity::setArrowColors(const std::string &colorAB, const std::string &colorPCP) {
    return MxCellPolarity_SetArrowColors(colorAB, colorPCP);
}

void CellPolarity::setArrowScale(const float &_scale) {
    return MxCellPolarity_SetArrowScale(_scale);
}

void CellPolarity::setArrowLength(const float &_length) {
    return MxCellPolarity_SetArrowLength(_length);
}

MxPolarityArrowData *CellPolarity::getVectorArrowAB(const int32_t &pId) {
    return MxCellPolarity_GetVectorArrowAB(pId);
}

MxPolarityArrowData *CellPolarity::getVectorArrowPCP(const int32_t &pId) {
    return MxCellPolarity_GetVectorArrowPCP(pId);
}

void CellPolarity::load() {
    return MxCellPolarity_load();
}

MxCellPolarityPotentialContact *CellPolarity::potentialContact(const float &cutoff, 
                                                               const float &mag, 
                                                               const float &rate, 
                                                               const float &distanceCoeff, 
                                                               const float &couplingFlat, 
                                                               const float &couplingOrtho, 
                                                               const float &couplingLateral, 
                                                               std::string contactType, 
                                                               const float &bendingCoeff) 
{
    return potential_create_cellpolarity(cutoff, mag, rate, distanceCoeff, couplingFlat, couplingOrtho, couplingLateral, contactType, bendingCoeff);
}
