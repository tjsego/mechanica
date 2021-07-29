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
#include "../../state/MxStateVector.h"

#include <algorithm>
#include <string>
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

void MxBoundaryCondition::set_potential(struct MxParticleType *ptype,
        struct MxPotential *pot)
{
    potenntials[ptype->id] = pot;
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
    left.set_potential(ptype, pot);
    right.set_potential(ptype, pot);
    front.set_potential(ptype, pot);
    back.set_potential(ptype, pot);
    bottom.set_potential(ptype, pot);
    top.set_potential(ptype, pot);
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
    switch(true);
    
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
    if(PyLong_Check(obj)) setValueAll(mx::cast<int>(obj));
    else if(PyDict_Check(obj)) {
        PyObject *keys = PyDict_Keys(obj);

        for(unsigned int i = 0; i < PyList_Size(keys); ++i) {
            PyObject *key = PyList_GetItem(keys, i);
            PyObject *value = PyDict_GetItem(obj, key);

            std::string name = mx::cast<std::string>(key);
            if(PyLong_Check(value)) {
                unsigned int v = mx::cast<unsigned int>(value);

                Log(LOG_DEBUG) << name << ": " << value;

                setValue(name, v);
            }
            else if(mx::check<std::string>(value)) {
                std::string s = mx::cast<std::string>(value);

                Log(LOG_DEBUG) << name << ": " << s;

                setValue(name, bc_kind_from_string(s));
            }
            else if(PyDict_Check(value)) {
                PyObject *vel = PyDict_GetItemString(value, "velocity");
                if(!vel) {
                    throw std::invalid_argument("attempt to initialize a boundary condition with a "
                                                "dictionary that does not contain a \'velocity\' item, "
                                                "only velocity boundary conditions support dictionary init");
                }
                MxVector3f v = mx::cast<MxVector3f>(vel);

                Log(LOG_DEBUG) << name << ": " << v;

                setVelocity(name, v);

                PyObject *restore = PyDict_GetItemString(value, "restore");
                if(restore) {
                    float r = mx::cast<float>(restore);

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
        
    if(bc.periodic & space_periodic_x &&
       src_cell->flags & cell_periodic_x &&
       dest_cell->flags & cell_periodic_x) {
        if(dest_cell->flags &  cell_periodic_left && bc.left.kind & BOUNDARY_RESETTING ) {
            if(p->state_vector) {
                p->state_vector->reset();
            }
            
        }
        
        if(dest_cell->flags &  cell_periodic_right && bc.right.kind & BOUNDARY_RESETTING ) {
            
        }
    }
    
    else if(_Engine.boundary_conditions.periodic & space_periodic_y &&
            src_cell->flags & cell_periodic_y &&
            dest_cell->flags & cell_periodic_y) {
        if(dest_cell->flags &  cell_periodic_front && bc.front.kind & BOUNDARY_RESETTING ) {
            
        }
        
        if(dest_cell->flags &  cell_periodic_back && bc.back.kind & BOUNDARY_RESETTING ) {
            
        }

    }
    
    else if(_Engine.boundary_conditions.periodic & space_periodic_z &&
            src_cell->flags & cell_periodic_z &&
            dest_cell->flags & cell_periodic_z) {
        if(dest_cell->flags &  cell_periodic_top && bc.top.kind & BOUNDARY_RESETTING ) {
            
        }
        
        if(dest_cell->flags &  cell_periodic_bottom && bc.bottom.kind & BOUNDARY_RESETTING ) {
            
        }

    }
}
