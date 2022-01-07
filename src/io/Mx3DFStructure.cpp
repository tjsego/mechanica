/**
 * @file Mx3DFStructure.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica 3D data format structure container
 * @date 2021-12-15
 * 
 */

#include <assimp/Importer.hpp>
#include <assimp/Exporter.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <Magnum/Trade/MeshData.h>
#include <Magnum/Primitives/Icosphere.h>

#include <MxLogger.h>
#include <mx_error.h>

#include <algorithm>
#include <limits>
#include <random>

#include "Mx3DFStructure.h"


namespace mx {

    MxMatrix4f cast(const aiMatrix4x4 &m) {
        return MxMatrix4f(
            {m.a1, m.a2, m.a3, m.a4}, 
            {m.b1, m.b2, m.b3, m.b4}, 
            {m.c1, m.c2, m.c3, m.c4}, 
            {m.d1, m.d2, m.d3, m.d4}
        );
    }

    aiMatrix4x4 cast(const MxMatrix4f &m) {
        return aiMatrix4x4(
            m[0][0], m[0][1], m[0][2], m[0][3], 
            m[1][0], m[1][1], m[1][2], m[1][3], 
            m[2][0], m[2][1], m[2][2], m[2][3], 
            m[3][0], m[3][1], m[3][2], m[3][3]
        );
    }

    MxVector3f cast(const aiVector3D &v) {
        return {v.x, v.y, v.z};
    }

    aiVector3D cast(const MxVector3f &v) {
        return {v.x(), v.y(), v.z()};
    }

};


const aiScene *getScene(Assimp::Importer &importer, const std::string &filePath, unsigned int *pFlags=NULL) {
    unsigned int _pFlags;
    if(pFlags == NULL) {
        _pFlags = aiProcess_ValidateDataStructure | 
                  aiProcess_JoinIdenticalVertices | 
                  aiProcess_GenNormals;
    }
    else _pFlags = *pFlags;

    return importer.ReadFile(filePath, _pFlags);
}


Mx3DFStructure::~Mx3DFStructure() {
    for(auto v : this->inventory.vertices) 
        this->remove(v);

    this->flush();
}

HRESULT getGlobalTransform(aiNode *node, aiMatrix4x4 *result) {
    if(node->mParent == NULL) 
        return S_OK;

    aiMatrix4x4 nt = aiMatrix4x4(node->mTransformation);
    getGlobalTransform(node->mParent, &nt);
    *result = *result * nt;
    return S_OK;
}

std::vector<unsigned int> buildVertexConsolidation(std::vector<MxVector3f> vPos, const float &tol=1E-6) {
    unsigned int numV = vPos.size();
    std::vector<unsigned int> result;
    result.reserve(numV);
    unsigned int consolidated = 0;

    for(unsigned int vIdx = 0; vIdx < numV; vIdx++) {
        result.push_back(vIdx);
        auto vp = vPos[vIdx];

        for(unsigned int vvIdx = 0; vvIdx < vIdx; vvIdx++) {
            if((vp - vPos[vvIdx]).length() < tol) {
                result[vIdx] = vvIdx;
                consolidated++;
                break;
            }
        }
    }

    Log(LOG_TRACE) << "Consolidated vertices: " << consolidated;

    return result;
}

/**
 * @brief Calculate face normals from vertices
 * 
 * @param vIndices 
 * @param vPos 
 * @return std::vector<MxVector3f> 
 */
MxVector3f faceNormal(std::vector<unsigned int> vIndices, std::vector<MxVector3f> vPos, std::vector<MxVector3f> vNorm) {
    MxVector3f vap = vPos[vIndices[0]],  vbp = vPos[vIndices[1]],  vcp = vPos[vIndices[2]];
    MxVector3f vbn = vNorm[vIndices[1]];

    MxVector3f result = Magnum::Math::cross(vcp - vbp, vap - vbp);
    float result_len = result.length();
    if(result_len == 0.f) 
        mx_error(E_FAIL, "Division by zero");
    result = result / result_len;

    // Check orientation

    if(result.dot(vbn) < 0) 
        result *= -1.0f;

    return result;
}

HRESULT Mx3DFStructure::fromFile(const std::string &filePath) {
    
    Log(LOG_DEBUG) << "Importing " << filePath;

    Assimp::Importer importer;
    auto scene = getScene(importer, filePath);
    if(scene == NULL) {
        std::string msg("Import failed (" + filePath + ")");
        mx_error(E_FAIL, msg.c_str());
    }

    if(!scene->HasMeshes()) { 
        Log(LOG_DEBUG) << "No meshes found";
        return S_OK;
    }

    // Get transforms to global frame
    // Either there is one root node, or a node hierarchy

    Log(LOG_DEBUG) << "Getting transforms";

    auto numNodes = scene->mRootNode->mNumChildren;
    auto numMeshes = scene->mNumMeshes;
    std::vector<MxMatrix4f> meshTransforms(scene->mNumMeshes);
    if(numNodes > 0) {
        Log(LOG_TRACE);

        for(unsigned int nIdx = 0; nIdx < numNodes; nIdx++) {
            auto node = scene->mRootNode->mChildren[nIdx];
            for(unsigned int nmIdx = 0; nmIdx < node->mNumMeshes; nmIdx++) {
                aiMatrix4x4 mt;
                getGlobalTransform(node, &mt);
                meshTransforms[node->mMeshes[nmIdx]] = mx::cast(mt);
            }
        }
    }
    else {
        Log(LOG_TRACE);

        for(unsigned int nmIdx = 0; nmIdx < scene->mNumMeshes; nmIdx++) {
            aiMatrix4x4 mt;
            getGlobalTransform(scene->mRootNode, &mt);
            meshTransforms[scene->mRootNode->mMeshes[nmIdx]] = mx::cast(mt);
        }
    }

    // Build data
    Log(LOG_DEBUG) << "Building data";

    for(unsigned int mIdx = 0; mIdx < numMeshes; mIdx++) {

        const aiMesh *aim = scene->mMeshes[mIdx];
        auto meshTransform = meshTransforms[mIdx];
        
        Log(LOG_TRACE) << "Mesh " << mIdx << ": " << aim->mNumVertices << " vertices, " << aim->mNumFaces << " faces";

        // Construct and add every vertex while keeping original ordering for future indexing
        Mx3DFVertexData *vertex;
        std::vector<Mx3DFVertexData*> vertices;
        std::vector<MxVector3f> vPos, vNorm;
        vertices.reserve(aim->mNumVertices);
        vPos.reserve(aim->mNumVertices);
        vNorm.reserve(aim->mNumVertices);
        for(unsigned int vIdx = 0; vIdx < aim->mNumVertices; vIdx++) {

            auto aiv = aim->mVertices[vIdx];
            auto ain = aim->mNormals[vIdx];
            
            MxVector4f ait = {(float)aiv.x, (float)aiv.y, (float)aiv.z, 1.f};
            MxVector3f position = (meshTransform * ait).xyz();
            vertex = new Mx3DFVertexData(position);

            vPos.push_back(position);

            MxVector3f aiu = {(float)ain.x, (float)ain.y, (float)ain.z};
            MxVector3f normal = meshTransform.rotation() * aiu;
            vNorm.push_back(normal);

            Log(LOG_TRACE) << "Vertex " << vIdx << ": " << position << ", " << normal;
            
            vertices.push_back(vertex);

        }

        // Consolidate

        auto vcIndices = buildVertexConsolidation(vPos);
        for(unsigned int vIdx = 0; vIdx < vPos.size(); vIdx++) 
            if(vcIndices[vIdx] == vIdx) 
                this->add(vertices[vIdx]);

        // If there are faces, construct them and their edges and add them
        if(aim->HasFaces()) {
            Log(LOG_TRACE) << "Importing " << std::to_string(aim->mNumFaces) << " faces";

            Mx3DFMeshData *mesh = new Mx3DFMeshData();
            mesh->name = std::string(aim->mName.C_Str());

            // Build and add faces and edges
            
            for(unsigned int fIdx = 0; fIdx < aim->mNumFaces; fIdx++) {
                auto aif = aim->mFaces[fIdx];

                std::vector<Mx3DFVertexData*> vertices_f;
                std::vector<unsigned int> fvIndices;
                vertices_f.reserve(aif.mNumIndices + 1);
                fvIndices.reserve(aif.mNumIndices);
                for(unsigned int fvIdx = 0; fvIdx < aif.mNumIndices; fvIdx++) { 
                    unsigned int vIdx = aif.mIndices[fvIdx];
                    vertices_f.push_back(vertices[vcIndices[vIdx]]);
                    fvIndices.push_back(vIdx);
                }

                vertices_f.push_back(vertices_f[0]);

                Mx3DFVertexData *va, *vb;
                Mx3DFEdgeData *edge;
                Mx3DFFaceData *face = new Mx3DFFaceData();
                for(unsigned int fvIdx = 0; fvIdx < aif.mNumIndices; fvIdx++) {
                    va = vertices_f[fvIdx];
                    vb = vertices_f[fvIdx + 1];
                    edge = NULL;
                    for(auto e : va->edges) 
                        if(e->has(vb)) {
                            edge = e;
                            break;
                        }
                    if(edge == NULL) {
                        edge = new Mx3DFEdgeData(va, vb);
                        this->add(edge);
                    }
                    
                    face->edges.push_back(edge);
                    edge->faces.push_back(face);
                }
                face->normal = faceNormal(fvIndices, vPos, vNorm);

                mesh->faces.push_back(face);
                face->meshes.push_back(mesh);
                this->add(face);

            }

            // Add final data for this mesh

            this->add(mesh);

        }

    }

    Log(LOG_INFORMATION) << "Successfully imported " << filePath;

    return S_OK;
}

std::vector<Mx3DFVertexData*> assembleFaceVertices(const std::vector<Mx3DFEdgeData*> &edges) {
    auto numEdges = edges.size();
    if(numEdges < 3) 
        mx_error(E_FAIL, "Invalid face definition from edges");
    auto numVertices = numEdges + 1;

    std::vector<bool> edgeIntegrated(edges.size(), false);
    std::vector<Mx3DFVertexData*> result(numVertices, 0);

    result[0] = edges[0]->va;
    result[1] = edges[0]->vb;
    edgeIntegrated[0] = true;

    Mx3DFVertexData *va = result[1];
    for(unsigned int vIdx = 2; vIdx < numVertices; vIdx++) {
        
        bool foundV = false;

        for(unsigned int eIdx = 0; eIdx < numEdges; eIdx++) {

            if(edgeIntegrated[eIdx]) 
                continue;

            auto e = edges[eIdx];
            if(e->va == va) { 
                foundV = true;
                edgeIntegrated[eIdx] = true;
                va = e->vb;
                result[vIdx] = va;
                break;
            } 
            else if(e->vb == va) { 
                foundV = true;
                edgeIntegrated[eIdx] = true;
                va = e->va;
                result[vIdx] = va;
                break;
            }

        }

        if(!foundV) 
            mx_error(E_FAIL, "Face assembly failed");
    }

    // Validate that last is first, then remove last

    if(result[0] != result[result.size() - 1]) 
        mx_error(E_FAIL, "Face result error");

    result.pop_back();

    return result;
}

HRESULT naiveNormalsCheck(std::vector<MxVector3f> &positions, std::vector<MxVector3f> &normals) {
    unsigned int i, numPos = positions.size();

    if(numPos != normals.size()) 
        mx_error(E_FAIL, "Positions and normals differ in size");

    // Build indices
    
    std::vector<unsigned int> iA, iB, iC;
    for(i = 0; i < positions.size(); i++) {

        iA.push_back(i-1); 
        iB.push_back(i); 
        iC.push_back(i+1);

    }
    iA[0] = positions.size() - 1;
    iC[positions.size() - 1] = 0;

    // Calculate norms and populate/correct as necessary

    MxVector3f pA, pB, pC, nB, nBCalc;
    bool flipIt = false;
    for(i = 0; i < positions.size() - 1; i++) {
        
        pA = positions[iA[i]]; 
        pB = positions[iB[i]]; 
        pC = positions[iC[i]];
        nB = normals[iB[i]];

        if(nB.length() < std::numeric_limits<float>::epsilon()) 
            normals[i] = nBCalc;
        else {
            nBCalc = Magnum::Math::cross(pC - pB, pA - pB);
            if(nB.dot(nBCalc) < 0) {
                flipIt = true;
                break;
            }
        }

    }

    // Do final correction: flip order of vertices if normals face inward

    if(flipIt) {

        std::vector<MxVector3f> tmp_p, tmp_n;

        for(i = 0; i < numPos; i++) {
            tmp_p.push_back(positions[numPos - i - 1]);
            tmp_n.push_back(normals[numPos - i - 1]);
        }
        for(i = 0; i < numPos; i++) {
            positions[i] = tmp_p[i];
            normals[i] = tmp_n[i];
        }

    }

    return S_OK;
}

void uploadMesh(aiMesh *aiMesh, Mx3DFMeshData *mxMesh, const unsigned int &mIdx=0) { 
    auto mxFaces = mxMesh->getFaces();
    auto mxEdges = mxMesh->getEdges();

    std::vector<std::vector<Mx3DFVertexData*> > verticesByFace;
    verticesByFace.reserve(mxFaces.size());

    aiMesh->mName = mxMesh->name.size() > 0 ? mxMesh->name : std::string("Mesh " + std::to_string(mIdx));
    aiMesh->mPrimitiveTypes = aiPrimitiveType_TRIANGLE | aiPrimitiveType_POLYGON;
    aiMesh->mNumFaces = mxFaces.size();

    Log(LOG_TRACE) << "... " << aiMesh->mNumFaces << " faces";

    unsigned int numVertices = 0;
    for(auto f : mxFaces) {
        
        auto fvertices = assembleFaceVertices(f->getEdges());
        verticesByFace.push_back(fvertices);
        numVertices += fvertices.size();

    }
    aiMesh->mNumVertices = numVertices;

    Log(LOG_TRACE) << "... prepping mesh";

    Log(LOG_TRACE) << "... " << aiMesh->mNumVertices << " vertices";

    // Allocate faces and vertices

    aiMesh->mFaces = new aiFace[aiMesh->mNumFaces];
    aiMesh->mVertices = new aiVector3D[aiMesh->mNumVertices];
    aiMesh->mNormals = new aiVector3D[aiMesh->mNumVertices];

    // Build faces and vertices; vertices take normals from their faces, if any

    for(unsigned int fIdx = 0, vIdx = 0; fIdx < aiMesh->mNumFaces; fIdx++) {

        aiFace &face = aiMesh->mFaces[fIdx];
        Mx3DFFaceData *mxFace = mxFaces[fIdx];
        
        auto fvertices = verticesByFace[fIdx];
        auto numfvertices = fvertices.size();

        face.mNumIndices = numfvertices;
        face.mIndices = new unsigned int[numfvertices];

        std::vector<MxVector3f> fpos, fnorm;
        for(unsigned int fvIdx = 0; fvIdx < numfvertices; fvIdx++) {
            fpos.push_back(fvertices[fvIdx]->position);
            fnorm.push_back(mxFaces[fIdx]->normal);
        }

        if(naiveNormalsCheck(fpos, fnorm) != S_OK) 
            return;

        for(unsigned int fvIdx = 0; fvIdx < numfvertices; fvIdx++, vIdx++) {
            face.mIndices[fvIdx] = vIdx;
            aiMesh->mVertices[vIdx] = mx::cast(fpos[fvIdx]);
            aiMesh->mNormals[vIdx] = mx::cast(fnorm[fvIdx]);
        }

    }
}

aiMaterial *generate3DFMaterial(Mx3DFRenderData *renderData) {

    if(renderData == NULL) 
        mx_error(E_FAIL, "NULL render data");

    aiMaterial *mtl = new aiMaterial();

    aiColor3D color = {renderData->color.x(), renderData->color.y(), renderData->color.z()};

    mtl->AddProperty(&color, 1, AI_MATKEY_COLOR_DIFFUSE);

    return mtl;
}

HRESULT Mx3DFStructure::toFile(const std::string &format, const std::string &filePath) {

    Log(LOG_DEBUG) << "Exporting " << format << ", " << filePath;

    Assimp::Exporter exporter;

    aiScene *scene = new aiScene();
    scene->mMetaData = new aiMetadata();

    // Create a root node with no child nodes

    aiNode *rootNode = new aiNode();

    // Set materials, if any; otherwise set one material

    // scene->mMaterials = new aiMaterial*[1];
    // scene->mNumMaterials = 1;
    // scene->mMaterials[0] = new aiMaterial();

    unsigned int numRenders = 0;
    auto mxmeshes = this->getMeshes();
    std::vector<aiMaterial*> meshMtls;
    std::vector<unsigned int> meshMtlIndices(mxmeshes.size(), 0);

    meshMtls.push_back(new aiMaterial());

    for(unsigned int i = 0; i < mxmeshes.size(); i++) {

        auto m = mxmeshes[i];
        if(m->renderData != NULL) {
            meshMtlIndices[i] = meshMtls.size();
            meshMtls.push_back(generate3DFMaterial(m->renderData));
        }

    }

    scene->mNumMaterials = meshMtls.size();
    scene->mMaterials = new aiMaterial*[scene->mNumMaterials];
    for(unsigned int i = 0; i < scene->mNumMaterials; i++) 
        scene->mMaterials[i] = meshMtls[i];

    // Create meshes

    scene->mNumMeshes = this->getNumMeshes();

    if(scene->mNumMeshes == 0) 
        mx_error(E_FAIL, "No data to export");

    Log(LOG_TRACE) << "number of meshes: " << scene->mNumMeshes;

    scene->mMeshes = new aiMesh*[scene->mNumMeshes];
    rootNode->mNumMeshes = scene->mNumMeshes;
    rootNode->mMeshes = new unsigned int[scene->mNumMeshes];
    for(unsigned int i = 0; i < scene->mNumMeshes; i++) { 
        scene->mMeshes[i] = new aiMesh();
        scene->mMeshes[i]->mMaterialIndex = meshMtlIndices[i];
        rootNode->mMeshes[i] = i;
    }
    scene->mRootNode = rootNode;

    for(unsigned int mIdx = 0; mIdx < scene->mNumMeshes; mIdx++) {
        Log(LOG_TRACE) << "generating mesh " << mIdx;
        
        uploadMesh(scene->mMeshes[mIdx], this->inventory.meshes[mIdx], mIdx);
    }

    Log(LOG_TRACE) << "Exporting";

    // Export

    if(exporter.Export(scene, format, filePath) != aiReturn_SUCCESS) {
        mx_error(E_FAIL, exporter.GetErrorString());
        return E_FAIL;
    }

    // Bug in MSVC: deleting a scene in debug builds invokes some issue with aiNode destructor
    #if !defined(_MSC_VER) || !defined(_DEBUG)
    delete scene;
    #endif

    return S_OK;
}

HRESULT Mx3DFStructure::flush() {
    
    for(auto v : this->queueRemove.vertices) 
        delete v;

    for(auto e : this->queueRemove.edges) 
        delete e;

    for(auto f : this->queueRemove.faces) 
        delete f;

    for(auto m : this->queueRemove.meshes) 
        delete m;

    this->queueRemove.vertices.clear();
    this->queueRemove.edges.clear();
    this->queueRemove.faces.clear();
    this->queueRemove.meshes.clear();

    return S_OK;
}

HRESULT Mx3DFStructure::extend(const Mx3DFStructure &s) {
    this->inventory.vertices.insert(this->inventory.vertices.end(), s.inventory.vertices.begin(), s.inventory.vertices.end());
    this->inventory.edges.insert(this->inventory.edges.end(),       s.inventory.edges.begin(),    s.inventory.edges.end());
    this->inventory.faces.insert(this->inventory.faces.end(),       s.inventory.faces.begin(),    s.inventory.faces.end());
    this->inventory.meshes.insert(this->inventory.meshes.end(),     s.inventory.meshes.begin(),   s.inventory.meshes.end());

    return S_OK;
}

HRESULT Mx3DFStructure::clear() {
    for(auto m : this->getMeshes()) 
        this->remove(m);

    for(auto f : this->getFaces()) 
        this->remove(f);

    for(auto e : this->getEdges()) 
        this->remove(e);

    for(auto v : this->getVertices()) 
        this->remove(v);

    return this->flush();
}

std::vector<Mx3DFVertexData*> Mx3DFStructure::getVertices() {
    return this->inventory.vertices;
}

std::vector<Mx3DFEdgeData*> Mx3DFStructure::getEdges() {
    return this->inventory.edges;
}

std::vector<Mx3DFFaceData*> Mx3DFStructure::getFaces() {
    return this->inventory.faces;
}

std::vector<Mx3DFMeshData*> Mx3DFStructure::getMeshes() {
    return this->inventory.meshes;
}

unsigned int Mx3DFStructure::getNumVertices() {
    return this->inventory.vertices.size();
}

unsigned int Mx3DFStructure::getNumEdges() {
    return this->inventory.edges.size();
}

unsigned int Mx3DFStructure::getNumFaces() {
    return this->inventory.faces.size();
}

unsigned int Mx3DFStructure::getNumMeshes() {
    return this->inventory.meshes.size();
}

bool Mx3DFStructure::has(Mx3DFVertexData *v) {
    auto itr = std::find(this->inventory.vertices.begin(), this->inventory.vertices.end(), v);
    return itr != std::end(this->inventory.vertices);
}

bool Mx3DFStructure::has(Mx3DFEdgeData *e) {
    auto itr = std::find(this->inventory.edges.begin(), this->inventory.edges.end(), e);
    return itr != std::end(this->inventory.edges);
}

bool Mx3DFStructure::has(Mx3DFFaceData *f) {
    auto itr = std::find(this->inventory.faces.begin(), this->inventory.faces.end(), f);
    return itr != std::end(this->inventory.faces);
}

bool Mx3DFStructure::has(Mx3DFMeshData *m) {
    auto itr = std::find(this->inventory.meshes.begin(), this->inventory.meshes.end(), m);
    return itr != std::end(this->inventory.meshes);
}

void Mx3DFStructure::add(Mx3DFVertexData *v) {
    if(v->structure == this) 
        return;
    else if(v->structure != NULL) 
        mx_error(E_FAIL, "Vertex already owned by a structure");

    v->structure = this;
    v->id = this->id_vertex++;
    this->inventory.vertices.push_back(v);
}

void Mx3DFStructure::add(Mx3DFEdgeData *e) {
    if(e->structure == this) 
        return;
    else if(e->structure != NULL) {
        std::stringstream msg_str;
        msg_str << "Edge already owned by a structure: ";
        msg_str << e->structure;
        mx_error(E_FAIL, msg_str.str().c_str());
    }
    else if(!e->va || !e->vb) 
        mx_error(E_FAIL, "Invalid definition");

    for(auto v : e->getVertices()) 
        if(v->structure != this) 
            this->add(v);

    e->structure = this;
    e->id = this->id_edge++;
    this->inventory.edges.push_back(e);
}

void Mx3DFStructure::add(Mx3DFFaceData *f) {
    if(f->structure == this) 
        return;
    else if(f->structure != NULL) 
        mx_error(E_FAIL, "Face already owned by a structure");
    else if(f->edges.size() < 3) 
        mx_error(E_FAIL, "Invalid definition");

    for(auto e : f->getEdges()) 
        if(e->structure != this) 
            this->add(e);

    f->structure = this;
    f->id = this->id_face++;
    this->inventory.faces.push_back(f);
}

void Mx3DFStructure::add(Mx3DFMeshData *m) {
    if(m->structure == this) 
        return;
    else if(m->structure != NULL) 
        mx_error(E_FAIL, "Mesh already owned by a structure");
    else if(m->faces.size() < 3) 
        mx_error(E_FAIL, "Invalid definition");

    for(auto f : m->faces) 
        if(f->structure != this) 
            this->add(f);

    m->structure = this;
    m->id = this->id_mesh++;
    this->inventory.meshes.push_back(m);
}

void Mx3DFStructure::remove(Mx3DFVertexData *v) {
    if(v->structure == NULL) 
        return;
    else if(v->structure != this) 
        mx_error(E_FAIL, "Vertex owned by different structure");

    auto itr = std::find(this->inventory.vertices.begin(), this->inventory.vertices.end(), v);
    if(itr == std::end(this->inventory.vertices)) 
        mx_error(E_FAIL, "Could not find vertex");
    
    this->onRemoved(v);

    v->structure = NULL;
    this->inventory.vertices.erase(itr);
    this->queueRemove.vertices.push_back(v);
}

void Mx3DFStructure::remove(Mx3DFEdgeData *e) {
    if(e->structure == NULL) 
        return;
    else if(e->structure != this) 
        mx_error(E_FAIL, "Edge owned by different structure");

    auto itr = std::find(this->inventory.edges.begin(), this->inventory.edges.end(), e);
    if(itr == std::end(this->inventory.edges)) 
        mx_error(E_FAIL, "Could not find edge");
    
    this->onRemoved(e);

    e->structure = NULL;
    this->inventory.edges.erase(itr);
    this->queueRemove.edges.push_back(e);
}

void Mx3DFStructure::remove(Mx3DFFaceData *f) {
    if(f->structure == NULL) 
        return;
    else if(f->structure != this) 
        mx_error(E_FAIL, "Face owned by different structure");

    auto itr = std::find(this->inventory.faces.begin(), this->inventory.faces.end(), f);
    if(itr == std::end(this->inventory.faces)) 
        mx_error(E_FAIL, "Could not find face");
    
    this->onRemoved(f);

    f->structure = NULL;
    this->inventory.faces.erase(itr);
    this->queueRemove.faces.push_back(f);
}

void Mx3DFStructure::remove(Mx3DFMeshData *m) {
    if(m->structure == NULL) 
        return;
    else if(m->structure != this) 
        mx_error(E_FAIL, "Mesh owned by different structure");

    auto itr = std::find(this->inventory.meshes.begin(), this->inventory.meshes.end(), m);
    if(itr == std::end(this->inventory.meshes)) 
        mx_error(E_FAIL, "Could not find mesh");
    
    m->structure = NULL;
    this->inventory.meshes.erase(itr);
    this->queueRemove.meshes.push_back(m);
}

void Mx3DFStructure::onRemoved(Mx3DFVertexData *v) {
    for(auto e : v->getEdges()) 
        this->remove(e);
}

void Mx3DFStructure::onRemoved(Mx3DFEdgeData *e) {
    for(auto f : e->getFaces()) { 
        auto itr = std::find(f->edges.begin(), f->edges.end(), e);
        f->edges.erase(itr);
        if(f->edges.size() < 3) 
            this->remove(f);
    }
}

void Mx3DFStructure::onRemoved(Mx3DFFaceData *f) {
    for(auto m : f->getMeshes()) {
        auto itr = std::find(m->faces.begin(), m->faces.end(), f);
        m->faces.erase(itr);
        if(m->faces.size() < 3) 
            this->remove(m);
    }
}

MxVector3f Mx3DFStructure::getCentroid() {
    auto vertices = this->getVertices();
    auto numV = vertices.size();

    if(numV == 0) 
        mx_error(E_FAIL, "No vertices");

    MxVector3f result = {0.f, 0.f, 0.f};

    for(unsigned int i = 0; i < numV; i++) 
        result += vertices[i]->position;

    result /= numV;
    return result;
}

HRESULT Mx3DFStructure::translate(const MxVector3f &displacement) {
    for(auto m : this->getMeshes()) 
        m->translate(displacement);

    return S_OK;
}

HRESULT Mx3DFStructure::translateTo(const MxVector3f &position) {
    return this->translate(position - this->getCentroid());
}

HRESULT Mx3DFStructure::rotateAt(const MxMatrix3f &rotMat, const MxVector3f &rotPt) {
    for(auto m : this->getMeshes()) 
        m->rotateAt(rotMat, rotPt);
    return S_OK;
}

HRESULT Mx3DFStructure::rotate(const MxMatrix3f &rotMat) {
    return this->rotateAt(rotMat, this->getCentroid());
}

HRESULT Mx3DFStructure::scaleFrom(const MxVector3f &scales, const MxVector3f &scalePt) {
    for(auto m : this->getMeshes()) {
        m->scaleFrom(scales, scalePt);
    }
    return S_OK;
}

HRESULT Mx3DFStructure::scaleFrom(const float &scale, const MxVector3f &scalePt) {
    return this->scaleFrom(MxVector3f(scale), scalePt);
}

HRESULT Mx3DFStructure::scale(const MxVector3f &scales) {
    return this->scaleFrom(scales, this->getCentroid());
}

HRESULT Mx3DFStructure::scale(const float &scale) {
    return this->scale(MxVector3f(scale));
}
