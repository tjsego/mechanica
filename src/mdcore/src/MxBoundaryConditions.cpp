/*
 * BoundaryConditions.cpp
 *
 *  Created on: Feb 10, 2021
 *      Author: andy
 */

#include "MxBoundaryConditions.hpp"
#include "space.h"
#include "engine.h"
#include "../../MxLogger.h"
#include "../../mx_error.h"
#include "../../io/MxFIO.h"
#include "../../state/MxStateVector.h"

#include <algorithm>
#include <string>
#include <cstring>
#include <unordered_map>

std::unordered_map<unsigned int, std::string> boundaryConditionsEnumToNameMap{
    {BOUNDARY_FREESLIP, "FREESLIP"},
    {BOUNDARY_NO_SLIP, "NOSLIP"},
    {BOUNDARY_PERIODIC, "PERIODIC"},
    {BOUNDARY_POTENTIAL, "POTENTIAL"},
    {BOUNDARY_RESETTING, "RESET"},
    {BOUNDARY_VELOCITY, "VELOCITY"}
};

std::unordered_map<std::string, unsigned int> boundaryConditionsNameToEnumMap{
    {"FREESLIP", BOUNDARY_FREESLIP},
    {"FREE_SLIP", BOUNDARY_FREESLIP},
    {"NOSLIP", BOUNDARY_NO_SLIP},
    {"NO_SLIP", BOUNDARY_NO_SLIP},
    {"PERIODIC", BOUNDARY_PERIODIC},
    {"POTENTIAL", BOUNDARY_POTENTIAL},
    {"RESET", BOUNDARY_RESETTING},
    {"VELOCITY", BOUNDARY_VELOCITY}
};

std::string invalidBoundaryConditionsName = "INVALID";


/**
 * boundary was initialized from flags, set individual values
 */
static void boundaries_from_flags(MxBoundaryConditions *bc) {
    
    if(bc->periodic & space_periodic_x) {
        bc->left.kind = BOUNDARY_PERIODIC;
        bc->right.kind = BOUNDARY_PERIODIC;
    }
    else if(bc->periodic & SPACE_FREESLIP_X) {
        bc->left.kind = BOUNDARY_FREESLIP;
        bc->right.kind = BOUNDARY_FREESLIP;
    }
    
    if(bc->periodic & space_periodic_y) {
        bc->front.kind = BOUNDARY_PERIODIC;
        bc->back.kind = BOUNDARY_PERIODIC;
    }
    else if(bc->periodic & SPACE_FREESLIP_Y) {
        bc->front.kind = BOUNDARY_FREESLIP;
        bc->back.kind = BOUNDARY_FREESLIP;
    }
    
    if(bc->periodic & space_periodic_z) {
        bc->top.kind = BOUNDARY_PERIODIC;
        bc->bottom.kind = BOUNDARY_PERIODIC;
    }
    else if(bc->periodic & SPACE_FREESLIP_Z) {
        bc->top.kind = BOUNDARY_FREESLIP;
        bc->bottom.kind = BOUNDARY_FREESLIP;
    }
}

// check if valid type, and return, if string and invalid string, throw exception.
static unsigned bc_kind_from_string(const std::string &s) {
    Log(LOG_DEBUG) << s;
    
    int result = 0;
    
    std::string _s = s;
    std::transform(_s.begin(), _s.end(), _s.begin(), ::toupper);

    auto itr = boundaryConditionsNameToEnumMap.find(_s);
    std::vector<std::string> validKindNames{
        "PERIODIC", "FREE_SLIP", "FREESLIP", "NO_SLIP", "NOSLIP", "POTENTIAL", "RESET"
    };
    if(itr!=boundaryConditionsNameToEnumMap.end()) {
        for (auto name : validKindNames) {
            if(_s.compare(name) == 0) {
                Log(LOG_DEBUG) << name;

                if(_s.compare("POTENTIAL") == 0) return itr->second | boundaryConditionsNameToEnumMap["FREESLIP"];
                return itr->second;
            }
        }
    }
    
    std::string msg = "invalid choice of value for boundary condition, \"" + _s + "\"";
    msg += ", only the following are supported for cardinal direction init: ";
    for(auto name : validKindNames) msg += "\"" + name + "\" ";
    mx_exp(std::invalid_argument(msg));
    return 0;
}

static unsigned int bc_kind_from_strings(const std::vector<std::string> &kinds) {
    Log(LOG_TRACE);

    int result = 0;

    for (auto k : kinds) result = result | bc_kind_from_string(k);
    
    return result;
}

static unsigned init_bc_direction(MxBoundaryCondition *low_bl, MxBoundaryCondition *high_bl, const unsigned &kind) {
    if(kind == BOUNDARY_NO_SLIP) {
        low_bl->kind = high_bl->kind = BOUNDARY_VELOCITY;
        low_bl->velocity = high_bl->velocity = MxVector3f{0.f, 0.f, 0.f};
    }
    else {
        low_bl->kind = (BoundaryConditionKind)kind;
        high_bl->kind = (BoundaryConditionKind)kind;
    }

    Log(LOG_DEBUG) << low_bl->name << ": " << low_bl->kindStr();
    Log(LOG_DEBUG) << high_bl->name << ": " << high_bl->kindStr();
    
    return kind;
}

unsigned MxBoundaryCondition::init(const unsigned &kind) {
    this->kind = (BoundaryConditionKind)kind;
    if(this->kind == BOUNDARY_NO_SLIP) {
        this->kind = BOUNDARY_VELOCITY;
        this->velocity = MxVector3f{};
    }

    Log(LOG_DEBUG) << this->name << ": " << kindStr();

    return this->kind;
}

unsigned MxBoundaryCondition::init(const MxVector3f &velocity, const float *restore) {
    if(restore) this->restore = *restore;
    this->kind = BOUNDARY_VELOCITY;
    this->velocity = velocity;

    Log(LOG_DEBUG) << this->name << ": " << kindStr();

    return this->kind;
}

unsigned MxBoundaryCondition::init(const std::unordered_map<std::string, unsigned int> vals, 
                                   const std::unordered_map<std::string, MxVector3f> vels, 
                                   const std::unordered_map<std::string, float> restores) 
{
    auto itr = vals.find(this->name);
    if(itr != vals.end()) return init(itr->second);

    auto itrv = vels.find(this->name);
    auto itrr = restores.find(this->name);
    if(itrv != vels.end()) {
        auto a = itrr == restores.end() ? NULL : &itrr->second;
        return init(itrv->second, a);
    }

    Log(LOG_DEBUG) << this->name << ": " << kindStr();

    return this->kind;
}

static void check_periodicy(MxBoundaryCondition *low_bc, MxBoundaryCondition *high_bc) {
    if((low_bc->kind & BOUNDARY_PERIODIC) ^ (high_bc->kind & BOUNDARY_PERIODIC)) {
        MxBoundaryCondition *has;
        MxBoundaryCondition *notHas;
    
        if(low_bc->kind & BOUNDARY_PERIODIC) {
            has = low_bc;
            notHas = high_bc;
        }
        else {
            has = high_bc;
            notHas = low_bc;
        }
        
        std::string msg = "only ";
        msg += has->name;
        msg += "has periodic boundary conditions set, but not ";
        msg += notHas->name;
        msg += ", setting both to periodic";
        
        low_bc->kind = BOUNDARY_PERIODIC;
        high_bc->kind = BOUNDARY_PERIODIC;
        
        Log(LOG_INFORMATION) << msg.c_str();
    }
}

MxBoundaryConditions::MxBoundaryConditions(int *cells) {
    if(_initIni() != S_OK) return;
    
	Log(LOG_INFORMATION) << "Initializing boundary conditions";

    this->periodic = space_periodic_full;
    boundaries_from_flags(this);
    
    _initFin(cells);
}

MxBoundaryConditions::MxBoundaryConditions(int *cells, const int &value) {
    if(_initIni() != S_OK) return;
    
    Log(LOG_INFORMATION) << "Initializing boundary conditions by value: " << value;
    
    switch(value) {
        case space_periodic_none :
        case space_periodic_x:
        case space_periodic_y:
        case space_periodic_z:
        case space_periodic_full:
        case space_periodic_ghost_x:
        case space_periodic_ghost_y:
        case space_periodic_ghost_z:
        case space_periodic_ghost_full:
        case SPACE_FREESLIP_X:
        case SPACE_FREESLIP_Y:
        case SPACE_FREESLIP_Z:
        case SPACE_FREESLIP_FULL:
            Log(LOG_INFORMATION) << "Processing as: SPACE_FREESLIP_FULL";

            this->periodic = value;
            break;
        default: {
            std::string msg = "invalid value " + std::to_string(value) + ", for integer boundary condition";
            mx_exp(std::invalid_argument(msg.c_str()));
            return;
        }
    }
    
    boundaries_from_flags(this);
    
    _initFin(cells);
}

void MxBoundaryConditions::_initDirections(const std::unordered_map<std::string, unsigned int> vals) {
    Log(LOG_INFORMATION) << "Initializing boundary conditions by directions";

    unsigned dir;
    auto itr = vals.find("x");
    if(itr != vals.end()) {
        dir = init_bc_direction(&(this->left), &(this->right), itr->second);
        if(dir & BOUNDARY_PERIODIC) {
            this->periodic |= space_periodic_x;
        }
        if(dir & BOUNDARY_FREESLIP) {
            this->periodic |= SPACE_FREESLIP_X;
        }
    }

    itr = vals.find("y");
    if(itr != vals.end()) {
        dir = init_bc_direction(&(this->front), &(this->back), itr->second);
        if(dir & BOUNDARY_PERIODIC) {
            this->periodic |= space_periodic_y;
        }
        if(dir & BOUNDARY_FREESLIP) {
            this->periodic |= SPACE_FREESLIP_Y;
        }
    }

    itr = vals.find("z");
    if(itr != vals.end()) {
        dir = init_bc_direction(&(this->bottom), &(this->top), itr->second);
        if(dir & BOUNDARY_PERIODIC) {
            this->periodic |= space_periodic_z;
        }
        if(dir & BOUNDARY_FREESLIP) {
            this->periodic |= SPACE_FREESLIP_Z;
        }
    }
}

void MxBoundaryConditions::_initSides(const std::unordered_map<std::string, unsigned int> vals, 
                                      const std::unordered_map<std::string, MxVector3f> vels, 
                                      const std::unordered_map<std::string, float> restores) 
{
    Log(LOG_INFORMATION) << "Initializing boundary conditions by sides";

    unsigned dir;

    dir = this->left.init(vals, vels, restores);
    if(dir & BOUNDARY_PERIODIC) {
        this->periodic |= space_periodic_x;
    }
    if(dir & BOUNDARY_FREESLIP) {
        this->periodic |= SPACE_FREESLIP_X;
    }

    dir = this->right.init(vals, vels, restores);
    if(dir & BOUNDARY_PERIODIC) {
        this->periodic |= space_periodic_x;
    }
    if(dir & BOUNDARY_FREESLIP) {
        this->periodic |= SPACE_FREESLIP_X;
    }

    dir = this->front.init(vals, vels, restores);
    if(dir & BOUNDARY_PERIODIC) {
        this->periodic |= space_periodic_y;
    }
    if(dir & BOUNDARY_FREESLIP) {
        this->periodic |= SPACE_FREESLIP_Y;
    }

    dir = this->back.init(vals, vels, restores);
    if(dir & BOUNDARY_PERIODIC) {
        this->periodic |= space_periodic_y;
    }
    if(dir & BOUNDARY_FREESLIP) {
        this->periodic |= SPACE_FREESLIP_Y;
    }

    dir = this->top.init(vals, vels, restores);
    if(dir & BOUNDARY_PERIODIC) {
        this->periodic |= space_periodic_z;
    }
    if(dir & BOUNDARY_FREESLIP) {
        this->periodic |= SPACE_FREESLIP_Z;
    }

    dir = this->bottom.init(vals, vels, restores);
    if(dir & BOUNDARY_PERIODIC) {
        this->periodic |= space_periodic_z;
    }
    if(dir & BOUNDARY_FREESLIP) {
        this->periodic |= SPACE_FREESLIP_Z;
    }
}

MxBoundaryConditions::MxBoundaryConditions(int *cells, 
                                           const std::unordered_map<std::string, unsigned int> vals, 
                                           const std::unordered_map<std::string, MxVector3f> vels, 
                                           const std::unordered_map<std::string, float> restores) 
{
    if(_initIni() != S_OK) return;

    Log(LOG_INFORMATION) << "Initializing boundary conditions by values";

    _initDirections(vals);
    _initSides(vals, vels, restores);

    check_periodicy(&(this->left), &(this->right));
    check_periodicy(&(this->front), &(this->back));
    check_periodicy(&(this->top), &(this->bottom));
    
    _initFin(cells);
}

// Initializes bc initialization, independently of what all was specified
HRESULT MxBoundaryConditions::_initIni() {
    Log(LOG_INFORMATION) << "Initializing boundary conditions initialization";

    bzero(this, sizeof(MxBoundaryConditions));

    this->potenntials = (MxPotential**)malloc(6 * engine::max_type * sizeof(MxPotential*));
    bzero(this->potenntials, 6 * engine::max_type * sizeof(MxPotential*));

    this->left.kind = BOUNDARY_PERIODIC;
    this->right.kind = BOUNDARY_PERIODIC;
    this->front.kind = BOUNDARY_PERIODIC;
    this->back.kind = BOUNDARY_PERIODIC;
    this->bottom.kind = BOUNDARY_PERIODIC;
    this->top.kind = BOUNDARY_PERIODIC;

    this->left.name = "left";     this->left.restore = 1.f;     this->left.potenntials =   &this->potenntials[0 * engine::max_type];
    this->right.name = "right";   this->right.restore = 1.f;    this->right.potenntials =  &this->potenntials[1 * engine::max_type];
    this->front.name = "front";   this->front.restore = 1.f;    this->front.potenntials =  &this->potenntials[2 * engine::max_type];
    this->back.name = "back";     this->back.restore = 1.f;     this->back.potenntials =   &this->potenntials[3 * engine::max_type];
    this->top.name = "top";       this->top.restore = 1.f;      this->top.potenntials =    &this->potenntials[4 * engine::max_type];
    this->bottom.name = "bottom"; this->bottom.restore = 1.f;   this->bottom.potenntials = &this->potenntials[5 * engine::max_type];
    
    this->left.normal =   { 1.f,  0.f,  0.f};
    this->right.normal =  {-1.f,  0.f,  0.f};
    this->front.normal =  { 0.f,  1.f,  0.f};
    this->back.normal =   { 0.f, -1.f,  0.f};
    this->bottom.normal = { 0.f,  0.f,  1.f};
    this->top.normal =    { 0.f,  0.f, -1.f};

    return S_OK;
}

// Finalizes bc initialization, independently of what all was specified
HRESULT MxBoundaryConditions::_initFin(int *cells) {
    Log(LOG_INFORMATION) << "Finalizing boundary conditions initialization";

    if(cells[0] < 3 && (this->periodic & space_periodic_x)) {
        cells[0] = 3;
        std::string msg = "requested periodic_x and " + std::to_string(cells[0]) +
        " space cells in the x direction, need at least 3 cells for periodic, setting cell count to 3";
        mx_exp(std::invalid_argument(msg.c_str()));
        return E_INVALIDARG;
    }
    if(cells[1] < 3 && (this->periodic & space_periodic_y)) {
        cells[1] = 3;
        std::string msg = "requested periodic_x and " + std::to_string(cells[1]) +
        " space cells in the x direction, need at least 3 cells for periodic, setting cell count to 3";
        mx_exp(std::invalid_argument(msg.c_str()));
        return E_INVALIDARG;
    }
    if(cells[2] < 3 && (this->periodic & space_periodic_z)) {
        cells[2] = 3;
        std::string msg = "requested periodic_x and " + std::to_string(cells[2]) +
        " space cells in the x direction, need at least 3 cells for periodic, setting cell count to 3";
        mx_exp(std::invalid_argument(msg.c_str()));
        return E_INVALIDARG;
    }
    
    Log(LOG_INFORMATION) << "engine periodic x : " << (bool)(this->periodic & space_periodic_x) ;
    Log(LOG_INFORMATION) << "engine periodic y : " << (bool)(this->periodic & space_periodic_y) ;
    Log(LOG_INFORMATION) << "engine periodic z : " << (bool)(this->periodic & space_periodic_z) ;
    Log(LOG_INFORMATION) << "engine freeslip x : " << (bool)(this->periodic & SPACE_FREESLIP_X) ;
    Log(LOG_INFORMATION) << "engine freeslip y : " << (bool)(this->periodic & SPACE_FREESLIP_Y) ;
    Log(LOG_INFORMATION) << "engine freeslip z : " << (bool)(this->periodic & SPACE_FREESLIP_Z) ;
    Log(LOG_INFORMATION) << "engine periodic ghost x : " << (bool)(this->periodic & space_periodic_ghost_x) ;
    Log(LOG_INFORMATION) << "engine periodic ghost y : " << (bool)(this->periodic & space_periodic_ghost_y) ;
    Log(LOG_INFORMATION) << "engine periodic ghost z : " << (bool)(this->periodic & space_periodic_ghost_z) ;
    
    return S_OK;
}

unsigned MxBoundaryConditions::boundaryKindFromString(const std::string &s) {
    return bc_kind_from_string(s);
}

std::string MxBoundaryConditions::str() {
    std::string s = "BoundaryConditions(\n";
    s += "  " + left.str(false) + ", \n";
    s += "  " + right.str(false) + ", \n";
    s += "  " + front.str(false) + ", \n";
    s += "  " + back.str(false) + ", \n";
    s += "  " + bottom.str(false) + ", \n";
    s += "  " + top.str(false) + ", \n";
    s += ")";
    return s;
}

#if defined(HAVE_CUDA)
static bool boundary_conditions_cuda_defer_update = false;
#endif

void MxBoundaryCondition::set_potential(struct MxParticleType *ptype,
        struct MxPotential *pot)
{
    potenntials[ptype->id] = pot;

    #if defined(HAVE_CUDA)
    if(!boundary_conditions_cuda_defer_update)
        engine_cuda_boundary_conditions_refresh(&_Engine);
    #endif
}

std::string MxBoundaryCondition::kindStr() const {
    std::string s = "";
    bool foundEntries = false;

    for(auto &itr : boundaryConditionsEnumToNameMap) {
        if(this->kind & itr.first) {
            if(foundEntries) s += ", " + itr.second;
            else s = itr.second;

            foundEntries = true;
        }
    }

    if(!foundEntries) s = invalidBoundaryConditionsName;

    return s;
}

std::string MxBoundaryCondition::str(bool show_type) const
{
    std::string s;
    
    if(show_type) {
        s +=  "BoundaryCondition(";
    }
    
    s += "\'";
    s += this->name;
    s += "\' : {";
    
    s += "\'kind\' : \'";
    s += kindStr();
    s += "\'";
    s += ", \'velocity\' : [" + std::to_string(velocity[0]) + ", " + std::to_string(velocity[1]) + ", " + std::to_string(velocity[2]) + "]";
    s += ", \'restore\' : " + std::to_string(restore);
    s += "}";
    
    if(show_type) {
        s +=  ")";
    }
    
    return s;
}

void MxBoundaryConditions::set_potential(struct MxParticleType *ptype,
        struct MxPotential *pot)
{
    #if defined(HAVE_CUDA)
    boundary_conditions_cuda_defer_update = true;
    #endif

    left.set_potential(ptype, pot);
    right.set_potential(ptype, pot);
    front.set_potential(ptype, pot);
    back.set_potential(ptype, pot);
    bottom.set_potential(ptype, pot);
    top.set_potential(ptype, pot);

    #if defined(HAVE_CUDA)
    boundary_conditions_cuda_defer_update = false;
    engine_cuda_boundary_conditions_refresh(&_Engine);
    #endif
}

void MxBoundaryConditionsArgsContainer::setValueAll(const int &_bcValue) {
    Log(LOG_INFORMATION) << std::to_string(_bcValue);
    
    switchType(true);
    *bcValue = _bcValue;
}

void MxBoundaryConditionsArgsContainer::setValue(const std::string &name, const unsigned int &value) {
    Log(LOG_INFORMATION) << name << ", " << std::to_string(value);

    switchType(false);
    (*bcVals)[name] = value;
}

void MxBoundaryConditionsArgsContainer::setVelocity(const std::string &name, const MxVector3f &velocity) {
    Log(LOG_INFORMATION) << name << ", " << std::to_string(velocity.x()) << ", " << std::to_string(velocity.y()) << ", " << std::to_string(velocity.z());

    switchType(false);
    (*bcVels)[name] = velocity;
}

void MxBoundaryConditionsArgsContainer::setRestore(const std::string &name, const float restore) {
    Log(LOG_INFORMATION) << name << ", " << std::to_string(restore);

    switchType(false);
    (*bcRestores)[name] = restore;
}

MxBoundaryConditions *MxBoundaryConditionsArgsContainer::create(int *cells) {
    MxBoundaryConditions *result;

    if(bcValue) {
        Log(LOG_INFORMATION) << "Creating boundary conditions by value."; 
        result = new MxBoundaryConditions(cells, *bcValue);
    }
    else if(bcVals) {
        Log(LOG_INFORMATION) << "Creating boundary conditions by values"; 
        result = new MxBoundaryConditions(cells, *bcVals, *bcVels, *bcRestores);
    }
    else {
        Log(LOG_INFORMATION) << "Creating boundary conditions by defaults";
        result = new MxBoundaryConditions(cells);
    }
    return result;
}

MxBoundaryConditionsArgsContainer::MxBoundaryConditionsArgsContainer(int *_bcValue, 
                                                                     std::unordered_map<std::string, unsigned int> *_bcVals, 
                                                                     std::unordered_map<std::string, MxVector3f> *_bcVels, 
                                                                     std::unordered_map<std::string, float> *_bcRestores) : 
    bcValue(nullptr), bcVals(nullptr), bcVels(nullptr), bcRestores(nullptr)
{
    if(_bcValue) setValueAll(*_bcValue);
    else {
        if(_bcVals)
            for(auto &itr : *_bcVals)
                setValue(itr.first, itr.second);
        if(_bcVels)
            for(auto &itr : *_bcVels)
                setVelocity(itr.first, itr.second);
        if(_bcRestores)
            for(auto &itr : *_bcRestores)
                setRestore(itr.first, itr.second);
    }
}

MxBoundaryConditionsArgsContainer::MxBoundaryConditionsArgsContainer(PyObject *obj) : 
    MxBoundaryConditionsArgsContainer()
{
    if(PyLong_Check(obj)) setValueAll(mx::cast<PyObject, int>(obj));
    else if(PyDict_Check(obj)) {
        PyObject *keys = PyDict_Keys(obj);

        for(unsigned int i = 0; i < PyList_Size(keys); ++i) {
            PyObject *key = PyList_GetItem(keys, i);
            PyObject *value = PyDict_GetItem(obj, key);

            std::string name = mx::cast<PyObject, std::string>(key);
            if(PyLong_Check(value)) {
                unsigned int v = mx::cast<PyObject, unsigned int>(value);

                Log(LOG_DEBUG) << name << ": " << value;

                setValue(name, v);
            }
            else if(mx::check<std::string>(value)) {
                std::string s = mx::cast<PyObject, std::string>(value);

                Log(LOG_DEBUG) << name << ": " << s;

                setValue(name, bc_kind_from_string(s));
            }
            else if(PySequence_Check(value)) {
                std::vector<std::string> kinds;
                PyObject *valueItem;
                for(unsigned int j = 0; j < PySequence_Size(value); j++) {
                    valueItem = PySequence_GetItem(value, j);
                    if(mx::check<std::string>(valueItem)) {
                        std::string s = mx::cast<PyObject, std::string>(valueItem);

                        Log(LOG_DEBUG) << name << ": " << s;

                        kinds.push_back(s);
                    }
                }
                setValue(name, bc_kind_from_strings(kinds));
            }
            else if(PyDict_Check(value)) {
                PyObject *vel = PyDict_GetItemString(value, "velocity");
                if(!vel) {
                    throw std::invalid_argument("attempt to initialize a boundary condition with a "
                                                "dictionary that does not contain a \'velocity\' item, "
                                                "only velocity boundary conditions support dictionary init");
                }
                MxVector3f v = mx::cast<PyObject, MxVector3f>(vel);

                Log(LOG_DEBUG) << name << ": " << v;

                setVelocity(name, v);

                PyObject *restore = PyDict_GetItemString(value, "restore");
                if(restore) {
                    float r = mx::cast<PyObject, float>(restore);

                    Log(LOG_DEBUG) << name << ": " << r;

                    setRestore(name, r);
                }
            }
        }

        Py_DECREF(keys);
    }
}

void MxBoundaryConditionsArgsContainer::switchType(const bool &allSides) {
    if(allSides) {
        if(bcVals) {
            delete bcVals;
            bcVals = NULL;
        }
        if(bcVels) {
            delete bcVels;
            bcVels = NULL;
        }
        if(bcRestores) {
            delete bcRestores;
            bcRestores = NULL;
        }

        if(!bcValue) bcValue = new int(BOUNDARY_PERIODIC);
    }
    else {
        if(bcValue) {
            delete bcValue;
            bcValue = NULL;
        }

        if(!bcVals) bcVals = new std::unordered_map<std::string, unsigned int>();
        if(!bcVels) bcVels = new std::unordered_map<std::string, MxVector3f>();
        if(!bcRestores) bcRestores = new std::unordered_map<std::string, float>();
    }
}

void apply_boundary_particle_crossing(struct MxParticle *p, const int *delta,
                                     const struct space_cell *src_cell, const struct space_cell *dest_cell) {
    
    const MxBoundaryConditions &bc = _Engine.boundary_conditions;

    if(!p->state_vector) 
        return;

    if(src_cell->loc[0] != dest_cell->loc[0]) {
        if(bc.periodic & space_periodic_x &&
            src_cell->flags & cell_periodic_x && dest_cell->flags & cell_periodic_x) 
        {
            
            if((dest_cell->flags  & cell_periodic_left  && bc.left.kind  & BOUNDARY_RESETTING) || 
                (dest_cell->flags & cell_periodic_right && bc.right.kind & BOUNDARY_RESETTING)) 
            {
                p->state_vector->reset();
                
            }
            
        }
    }

    else if(src_cell->loc[1] != dest_cell->loc[1]) {
        if(bc.periodic & space_periodic_y &&
            src_cell->flags & cell_periodic_y && dest_cell->flags & cell_periodic_y) 
        {
            
            if((dest_cell->flags  & cell_periodic_front && bc.front.kind & BOUNDARY_RESETTING) || 
                (dest_cell->flags & cell_periodic_back  && bc.back.kind  & BOUNDARY_RESETTING)) 
            {
                p->state_vector->reset();
                
            }

        }
    } 
    
    else if(src_cell->loc[2] != dest_cell->loc[2]) {
        if(bc.periodic & space_periodic_z &&
            src_cell->flags & cell_periodic_z && dest_cell->flags & cell_periodic_z) 
        {
            
            if((dest_cell->flags  & cell_periodic_top    && bc.top.kind    & BOUNDARY_RESETTING) ||
                (dest_cell->flags & cell_periodic_bottom && bc.bottom.kind & BOUNDARY_RESETTING)) 
            {
                p->state_vector->reset();
                
            }

        }
    }
}

std::string MxBoundaryConditions::toString() {
    return mx::io::toString(*this);
}

MxBoundaryConditions *MxBoundaryConditions::fromString(const std::string &str) {
    return new MxBoundaryConditions(mx::io::fromString<MxBoundaryConditions>(str));
}


namespace mx { namespace io {

#define MXBOUNDARYCONDIOTOEASY(fe, key, member) \
    fe = new MxIOElement(); \
    if(toFile(member, metaData, fe) != S_OK)  \
        return E_FAIL; \
    fe->parent = fileElement; \
    fileElement->children[key] = fe;

#define MXBOUNDARYCONDIOFROMEASY(feItr, children, metaData, key, member_p) \
    feItr = children.find(key); \
    if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
        return E_FAIL;

template <>
HRESULT toFile(const MxBoundaryCondition &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    MxIOElement *fe;

    MXBOUNDARYCONDIOTOEASY(fe, "kind", (int)dataElement.kind);
    MXBOUNDARYCONDIOTOEASY(fe, "id", dataElement.id);
    if(dataElement.kind & BOUNDARY_VELOCITY) {
        MXBOUNDARYCONDIOTOEASY(fe, "velocity", dataElement.velocity);
    }
    MXBOUNDARYCONDIOTOEASY(fe, "restore", dataElement.restore);
    MXBOUNDARYCONDIOTOEASY(fe, "name", std::string(dataElement.name));
    MXBOUNDARYCONDIOTOEASY(fe, "normal", dataElement.normal);

    std::vector<unsigned int> potentialIndices;
    std::vector<MxPotential*> potentials;
    MxPotential *pot;
    for(unsigned int i = 0; i < engine::max_type; i++) {
        pot = dataElement.potenntials[i];
        if(pot != NULL) {
            potentialIndices.push_back(i);
            potentials.push_back(pot);
        }
    }
    if(potentialIndices.size() > 0) {
        MXBOUNDARYCONDIOTOEASY(fe, "potentialIndices", potentialIndices);
        MXBOUNDARYCONDIOTOEASY(fe, "potentials", potentials);
    }

    MXBOUNDARYCONDIOTOEASY(fe, "radius", dataElement.radius);

    fileElement->type = "boundaryCondition";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxBoundaryCondition *dataElement) {

    MxIOChildMap::const_iterator feItr;

    unsigned int kind;
    MXBOUNDARYCONDIOFROMEASY(feItr, fileElement.children, metaData, "kind", &kind);
    dataElement->kind = (BoundaryConditionKind)kind;

    MXBOUNDARYCONDIOFROMEASY(feItr, fileElement.children, metaData, "id", &dataElement->id);

    if(fileElement.children.find("velocity") != fileElement.children.end()) {
        MxVector3f velocity;
        MXBOUNDARYCONDIOFROMEASY(feItr, fileElement.children, metaData, "velocity", &velocity);
        dataElement->velocity = velocity;
    }

    MXBOUNDARYCONDIOFROMEASY(feItr, fileElement.children, metaData, "restore", &dataElement->restore);

    std::string name;
    MXBOUNDARYCONDIOFROMEASY(feItr, fileElement.children, metaData, "name", &name);
    char *cname = new char[name.size() + 1];
	std::strcpy(cname, name.c_str());
	dataElement->name = cname;

    MXBOUNDARYCONDIOFROMEASY(feItr, fileElement.children, metaData, "normal", &dataElement->normal);

    if(fileElement.children.find("potentials") != fileElement.children.end()) {

        std::vector<unsigned int> potentialIndices;
        std::vector<MxPotential*> potentials;
        MXBOUNDARYCONDIOFROMEASY(feItr, fileElement.children, metaData, "potentialIndices", &potentialIndices);
        MXBOUNDARYCONDIOFROMEASY(feItr, fileElement.children, metaData, "potentials", &potentials);

        if(potentials.size() > 0) 
            for(unsigned int i = 0; i < potentials.size(); i++) 
                dataElement->potenntials[potentialIndices[i]] = potentials[i];

    }

    MXBOUNDARYCONDIOFROMEASY(feItr, fileElement.children, metaData, "radius", &dataElement->radius);

    return S_OK;
}

template <>
HRESULT toFile(const MxBoundaryConditions &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    MxIOElement *fe;

    MXBOUNDARYCONDIOTOEASY(fe, "top", dataElement.top);
    MXBOUNDARYCONDIOTOEASY(fe, "bottom", dataElement.bottom);
    MXBOUNDARYCONDIOTOEASY(fe, "left", dataElement.left);
    MXBOUNDARYCONDIOTOEASY(fe, "right", dataElement.right);
    MXBOUNDARYCONDIOTOEASY(fe, "front", dataElement.front);
    MXBOUNDARYCONDIOTOEASY(fe, "back", dataElement.back);

    fileElement->type = "boundaryConditions";
    
    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxBoundaryConditions *dataElement) {

    MxIOChildMap::const_iterator feItr;

    // Initialize potential arrays
    // todo: implement automatic initialization of potential arrays in boundary conditions under all circumstances

    dataElement->potenntials = (MxPotential**)malloc(6 * engine::max_type * sizeof(MxPotential*));
    bzero(dataElement->potenntials, 6 * engine::max_type * sizeof(MxPotential*));

    dataElement->left.potenntials =   &dataElement->potenntials[0 * engine::max_type];
    dataElement->right.potenntials =  &dataElement->potenntials[1 * engine::max_type];
    dataElement->front.potenntials =  &dataElement->potenntials[2 * engine::max_type];
    dataElement->back.potenntials =   &dataElement->potenntials[3 * engine::max_type];
    dataElement->top.potenntials =    &dataElement->potenntials[4 * engine::max_type];
    dataElement->bottom.potenntials = &dataElement->potenntials[5 * engine::max_type];

    MXBOUNDARYCONDIOFROMEASY(feItr, fileElement.children, metaData, "top",    &dataElement->top);
    MXBOUNDARYCONDIOFROMEASY(feItr, fileElement.children, metaData, "bottom", &dataElement->bottom);
    MXBOUNDARYCONDIOFROMEASY(feItr, fileElement.children, metaData, "left",   &dataElement->left);
    MXBOUNDARYCONDIOFROMEASY(feItr, fileElement.children, metaData, "right",  &dataElement->right);
    MXBOUNDARYCONDIOFROMEASY(feItr, fileElement.children, metaData, "front",  &dataElement->front);
    MXBOUNDARYCONDIOFROMEASY(feItr, fileElement.children, metaData, "back",   &dataElement->back);

    return S_OK;
}

#define MXBOUNDARYCONDARGSIOFROMEASY(side) \
    feItr = fileElement.children.find(side); \
    if(feItr == fileElement.children.end())  \
        return E_FAIL; \
    fe = feItr->second; \
    MXBOUNDARYCONDIOFROMEASY(feItr, fe->children, metaData, "kind", &kind); \
    bcVals[side] = kind; \
    MXBOUNDARYCONDIOFROMEASY(feItr, fe->children, metaData, "restore", &restore); \
    bcRestores[side] = restore; \
    if((BoundaryConditionKind)kind & BOUNDARY_VELOCITY) { \
        MXBOUNDARYCONDIOFROMEASY(feItr, fe->children, metaData, "velocity", &velocity); \
        bcVels["velocity"] = velocity; \
    }

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxBoundaryConditionsArgsContainer *dataElement) {

    MxIOChildMap::const_iterator feItr;
    MxIOElement *fe;
    unsigned int kind;
    std::string side;
    MxVector3f velocity;
    float restore;

    std::unordered_map<std::string, unsigned int> bcVals; 
    std::unordered_map<std::string, MxVector3f> bcVels; 
    std::unordered_map<std::string, float> bcRestores;

    MXBOUNDARYCONDARGSIOFROMEASY("top");
    MXBOUNDARYCONDARGSIOFROMEASY("bottom");
    MXBOUNDARYCONDARGSIOFROMEASY("left");
    MXBOUNDARYCONDARGSIOFROMEASY("right");
    MXBOUNDARYCONDARGSIOFROMEASY("front");
    MXBOUNDARYCONDARGSIOFROMEASY("back");

    dataElement = new MxBoundaryConditionsArgsContainer(0, &bcVals, &bcVels, &bcRestores);

    return S_OK;
}

}};
