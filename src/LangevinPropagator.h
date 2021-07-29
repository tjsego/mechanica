/*
 * MeshDampedLangevinPropagator.h
 *
 *  Created on: Aug 3, 2017
 *      Author: andy
 */

#ifndef SRC_MESHDAMPEDLANGEVINPROPAGATOR_H_
#define SRC_MESHDAMPEDLANGEVINPROPAGATOR_H_

#include "MxPropagator.h"
#include "Magnum/Magnum.h"
#include "MxConstraints.h"
#include "MxForces.h"

struct MxModel;
struct MxMesh;



/**
 * Damped Langevin propagator,
 *
 * Calculates time evolution via the over-damped Langevin equation,
 *
 * m dx/dt = F(x)/ \gamma + \eta(t)
 */
class LangevinPropagator {

    typedef MxVector3f Vector3;



public:

    LangevinPropagator();
    
    /**
     * Attaches model to this propagator
     */
    HRESULT setModel(MxModel *model);

    HRESULT step(MxReal dt);

    /**
     * Inform the propagator that the model structure changed.
     */
    HRESULT structureChanged();


    HRESULT bindConstraint(IConstraint *constraint, MxConstrainableType *obj);

    HRESULT bindForce(IForce *force, MxForcableType *obj);

    HRESULT unbindConstraint(IConstraint* constraint);

    HRESULT unbindForce(IForce *force);



private:

    template<typename ActorType, typename TargetType, typename TargetObject>
    struct BaseItems {
        BaseItems(ActorType *actor, TargetType *type=NULL) {
            this->actor = actor; 
            this->type = type;
        }
        virtual void update(const LangevinPropagator *prop) {}
        void unbind() {
            type = NULL;
            args.clear();
        }

        ActorType *actor;
        TargetType *type;
        std::vector<TargetObject*> args;
    };
    
    struct _ConstraintItems : BaseItems<IConstraint, MxConstrainableType, MxConstrainable> {
        _ConstraintItems(IConstraint *actor, MxConstrainableType *type=NULL) : BaseItems<IConstraint, MxConstrainableType, MxConstrainable>(actor, type) {}
        void update(const LangevinPropagator *prop);
    };
    using ConstraintItems = BaseItems<IConstraint, MxConstrainableType, MxConstrainable>;

    struct _ForceItems : BaseItems<IForce, MxForcableType, MxForcable> {
        _ForceItems(IForce *actor, MxForcableType *type=NULL) : BaseItems<IForce, MxForcableType, MxForcable>(actor, type) {}
        void update(const LangevinPropagator *prop);
    };
    using ForceItems = BaseItems<IForce, MxForcableType, MxForcable>;

    HRESULT applyForces();


    HRESULT eulerStep(MxReal dt);

    HRESULT rungeKuttaStep(MxReal dt);


    HRESULT getAccelerations(float time, uint32_t len, const Vector3 *pos, Vector3 *acc);

    //HRESULT getMasses(float time, uint32_t len, float *masses);

    HRESULT getPositions(float time, uint32_t len, Vector3 *pos);

    HRESULT setPositions(float time, uint32_t len, const Vector3 *pos);

    HRESULT applyConstraints();
    
    /**
     * The model structure changed, so we need to update all the
     * constraints
     */



    MxModel *model;
    MxMesh *mesh;

    size_t size = 0;
    Vector3 *positions = nullptr;

    Vector3 *posInit = nullptr;

    Vector3 *accel = nullptr;

    Vector3 *k1 = nullptr;
    Vector3 *k2 = nullptr;
    Vector3 *k3 = nullptr;
    Vector3 *k4 = nullptr;

    float *masses = nullptr;

    void resize();

    size_t timeSteps = 0;

    uint32_t stateVectorSize = 0;
    float *stateVectorInit = nullptr;
    float *stateVector = nullptr;

    float *stateVectorK1 = nullptr;
    float *stateVectorK2 = nullptr;
    float *stateVectorK3 = nullptr;
    float *stateVectorK4 = nullptr;

    HRESULT stateVectorStep(MxReal dt);


    /**
     * Keep track of constrained objects, most objects aren't constrained.
     */
    std::vector<ConstraintItems> constraints;
    std::vector<ForceItems> forces;

    template<typename A, typename T, typename O>
    HRESULT updateItems(std::vector<LangevinPropagator::BaseItems<A, T, O> > &items);

    template<typename A, typename T, typename O>
    LangevinPropagator::BaseItems<A, T, O>& getItem(std::vector<LangevinPropagator::BaseItems<A, T, O> > &items, A *key);

    template<typename A, typename T, typename O>
    HRESULT bindTypeItem(std::vector<LangevinPropagator::BaseItems<A, T, O> > &items, A *key, T* type);

};

HRESULT MxBind_PropagatorModel(LangevinPropagator *propagator, MxModel *model);



#endif /* SRC_MESHDAMPEDLANGEVINPROPAGATOR_H_ */
