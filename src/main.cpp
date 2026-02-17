#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <memory>
#include <climits>
#include <cfloat>
#include <algorithm>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "GLSL.h"
#include "MatrixStack.h"

using namespace std;
using namespace glm;

struct MeshData {
    GLuint posBuf;
    GLuint norBuf;
    int count;
};

class Component {
public:
    string name;
    vec3 jointTranslate;
    vec3 jointAngles;

    vec3 animatedAngles;
    bool animate;

    vec3 meshTranslate;
    vec3 meshScale;
    vector<Component*> children;

    Component(string n)
        : name(n),
        jointTranslate(0.0f),
        jointAngles(0.0f),
        animatedAngles(0.0f),
        animate(false),
        meshTranslate(0.0f),
        meshScale(1.0f) {
    }

    void flatten(vector<Component*>& list) {
        list.push_back(this);
        for (auto child : children) child->flatten(list);
    }

    void draw(MatrixStack& MV, GLint unifMV,
        GLint attrPos, GLint attrNor,
        MeshData& cube, MeshData& sphere,
        Component* selected,
        float pulsateBase, float pulsateFreq) { // <- use constants
        MV.pushMatrix();

        MV.translate(jointTranslate);
        MV.rotate(jointAngles.x, vec3(1, 0, 0));
        MV.rotate(jointAngles.y, vec3(0, 1, 0));
        MV.rotate(jointAngles.z, vec3(0, 0, 1));

        if (animate) {
            MV.pushMatrix();
            MV.rotate(animatedAngles.x, vec3(1, 0, 0));
            MV.rotate(animatedAngles.y, vec3(0, 1, 0));
            MV.rotate(animatedAngles.z, vec3(0, 0, 1));
        }

        // --- Draw joint sphere ---
        MV.pushMatrix();
        MV.scale(vec3(0.25f));
        glUniformMatrix4fv(unifMV, 1, GL_FALSE, value_ptr(MV.topMatrix()));
        glBindBuffer(GL_ARRAY_BUFFER, sphere.posBuf);
        glVertexAttribPointer(attrPos, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glBindBuffer(GL_ARRAY_BUFFER, sphere.norBuf);
        glVertexAttribPointer(attrNor, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glDrawArrays(GL_TRIANGLES, 0, sphere.count);
        MV.popMatrix();

        // --- Draw cube (limb) ---
        MV.pushMatrix();
        MV.translate(meshTranslate);

        float s_pulsate = 1.0f;
        if (this == selected) {
            double t = glfwGetTime();
            s_pulsate = 1.0f + pulsateBase + pulsateBase * sin(pulsateFreq * (float)t);
        }

        MV.scale(meshScale * s_pulsate);
        glUniformMatrix4fv(unifMV, 1, GL_FALSE, value_ptr(MV.topMatrix()));
        glBindBuffer(GL_ARRAY_BUFFER, cube.posBuf);
        glVertexAttribPointer(attrPos, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glBindBuffer(GL_ARRAY_BUFFER, cube.norBuf);
        glVertexAttribPointer(attrNor, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glDrawArrays(GL_TRIANGLES, 0, cube.count);
        MV.popMatrix();

        if (animate) MV.popMatrix();

        for (auto child : children)
            child->draw(MV, unifMV, attrPos, attrNor, cube, sphere, selected, pulsateBase, pulsateFreq);

        MV.popMatrix();
    }
};

// ============================================================
// Globals
// ============================================================
GLFWwindow* window;
string RESOURCE_DIR = "./";
GLuint progID;
map<string, GLint> attrIDs;
map<string, GLint> unifIDs;
MeshData cubeMesh, sphereMesh;
Component* robotRoot = nullptr;
Component* selectedComponent = nullptr;
vector<Component*> componentList;
int selectionIndex = 0;

// ==== Animation / Pulsate Globals ====
float SPIN_SPEED = 3.0f;
float PULSATE_BASE = 0.025f;
float PULSATE_FREQ = 4.0f;

// ==== TASK 7 Globals ====
vec2 mousePrev = vec2(0.0f, 0.0f);
bool firstMouse = true;

// ============================================================
// Callbacks
// ============================================================
static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        vec2 mouseCurr = vec2((float)xpos, (float)ypos);

        if (firstMouse) {
            mousePrev = mouseCurr;
            firstMouse = false;
        }

        vec2 delta = mouseCurr - mousePrev;
        float sensitivity = 0.01f;

        if (robotRoot) {
            robotRoot->jointAngles.y += delta.x * sensitivity;
            robotRoot->jointAngles.x += delta.y * sensitivity;
        }

        mousePrev = mouseCurr;
    }
    else {
        firstMouse = true;
    }
}

static void char_callback(GLFWwindow* window, unsigned int key) {
    float step = 0.1f;
    if (key == '.') selectionIndex = (selectionIndex + 1) % (int)componentList.size();
    if (key == ',') selectionIndex = (selectionIndex - 1 + (int)componentList.size()) % (int)componentList.size();
    selectedComponent = componentList[selectionIndex];

    if (key == 'x') selectedComponent->jointAngles.x += step;
    if (key == 'X') selectedComponent->jointAngles.x -= step;
    if (key == 'y') selectedComponent->jointAngles.y += step;
    if (key == 'Y') selectedComponent->jointAngles.y -= step;
    if (key == 'z') selectedComponent->jointAngles.z += step;
    if (key == 'Z') selectedComponent->jointAngles.z -= step;
}

// ============================================================
// Utilities
// ============================================================
void createSphere(MeshData& mesh) {
    vector<float> p, n;
    int stacks = 20, slices = 20;
    for (int i = 0; i < stacks; ++i) {
        float phi1 = pi<float>() * i / stacks;
        float phi2 = pi<float>() * (i + 1) / stacks;
        for (int j = 0; j < slices; ++j) {
            float t1 = 2.f * pi<float>() * j / slices;
            float t2 = 2.f * pi<float>() * (j + 1) / slices;
            auto addV = [&](float pv, float tv) {
                float x = sin(pv) * cos(tv); float y = cos(pv); float z = sin(pv) * sin(tv);
                p.push_back(x); p.push_back(y); p.push_back(z);
                n.push_back(x); n.push_back(y); n.push_back(z);
                };
            addV(phi1, t1); addV(phi2, t1); addV(phi1, t2);
            addV(phi1, t2); addV(phi2, t1); addV(phi2, t2);
        }
    }
    mesh.count = p.size() / 3;
    glGenBuffers(1, &mesh.posBuf); glBindBuffer(GL_ARRAY_BUFFER, mesh.posBuf); glBufferData(GL_ARRAY_BUFFER, p.size() * sizeof(float), p.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &mesh.norBuf); glBindBuffer(GL_ARRAY_BUFFER, mesh.norBuf); glBufferData(GL_ARRAY_BUFFER, n.size() * sizeof(float), n.data(), GL_STATIC_DRAW);
}

static void render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    int w, h; glfwGetFramebufferSize(window, &w, &h);
    if (w == 0 || h == 0) return;

    mat4 P = perspective(radians(45.f), (float)w / h, 0.01f, 100.f);
    glUseProgram(progID);
    glUniformMatrix4fv(unifIDs["P"], 1, GL_FALSE, value_ptr(P));
    glEnableVertexAttribArray(attrIDs["aPos"]);
    glEnableVertexAttribArray(attrIDs["aNor"]);

    MatrixStack MV;
    MV.pushMatrix();
    MV.translate(vec3(0, 0, -10));
    if (robotRoot)
        robotRoot->draw(MV, unifIDs["MV"], attrIDs["aPos"], attrIDs["aNor"], cubeMesh, sphereMesh, selectedComponent, PULSATE_BASE, PULSATE_FREQ);
    MV.popMatrix();
}

static void init() {
    glClearColor(1, 1, 1, 1); glEnable(GL_DEPTH_TEST);
    char* vS = GLSL::textFileRead((RESOURCE_DIR + "vert.glsl").c_str());
    char* fS = GLSL::textFileRead((RESOURCE_DIR + "frag.glsl").c_str());
    GLuint vID = glCreateShader(GL_VERTEX_SHADER); GLuint fID = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(vID, 1, &vS, NULL); glShaderSource(fID, 1, &fS, NULL);
    glCompileShader(vID); glCompileShader(fID);
    progID = glCreateProgram(); glAttachShader(progID, vID); glAttachShader(progID, fID); glLinkProgram(progID);
    attrIDs["aPos"] = glGetAttribLocation(progID, "aPos"); attrIDs["aNor"] = glGetAttribLocation(progID, "aNor");
    unifIDs["P"] = glGetUniformLocation(progID, "P"); unifIDs["MV"] = glGetUniformLocation(progID, "MV");

    tinyobj::attrib_t attr; vector<tinyobj::shape_t> shp; vector<tinyobj::material_t> mat; string w, e;
    tinyobj::LoadObj(&attr, &shp, &mat, &w, &e, (RESOURCE_DIR + "cube.obj").c_str(), NULL, true, true);
    vector<float> pB, nB;
    for (auto& s : shp) for (auto& idx : s.mesh.indices) {
        pB.push_back(attr.vertices[3 * idx.vertex_index + 0]); pB.push_back(attr.vertices[3 * idx.vertex_index + 1]); pB.push_back(attr.vertices[3 * idx.vertex_index + 2]);
        nB.push_back(attr.normals[3 * idx.normal_index + 0]); nB.push_back(attr.normals[3 * idx.normal_index + 1]); nB.push_back(attr.normals[3 * idx.normal_index + 2]);
    }
    cubeMesh.count = pB.size() / 3;
    glGenBuffers(1, &cubeMesh.posBuf); glBindBuffer(GL_ARRAY_BUFFER, cubeMesh.posBuf); glBufferData(GL_ARRAY_BUFFER, pB.size() * sizeof(float), pB.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &cubeMesh.norBuf); glBindBuffer(GL_ARRAY_BUFFER, cubeMesh.norBuf); glBufferData(GL_ARRAY_BUFFER, nB.size() * sizeof(float), nB.data(), GL_STATIC_DRAW);

    createSphere(sphereMesh);

    // ---- Robot hierarchy ----
    robotRoot = new Component("Torso");
    robotRoot->meshScale = vec3(1.2f, 1.8f, 0.6f);

    Component* head = new Component("Head");
    head->jointTranslate = vec3(0, 0.9f, 0); head->meshTranslate = vec3(0, 0.25f, 0); head->meshScale = vec3(0.5f);
    robotRoot->children.push_back(head);

    Component* LUArm = new Component("L_UpperArm");
    LUArm->jointTranslate = vec3(-0.6f, 0.7f, 0); LUArm->meshTranslate = vec3(-0.4f, 0, 0); LUArm->meshScale = vec3(0.8f, 0.25f, 0.25f);
    robotRoot->children.push_back(LUArm);

    Component* LLArm = new Component("L_LowerArm");
    LLArm->jointTranslate = vec3(-0.8f, 0, 0); LLArm->meshTranslate = vec3(-0.35f, 0, 0); LLArm->meshScale = vec3(0.7f, 0.2f, 0.2f); LLArm->animate = true;
    LUArm->children.push_back(LLArm);

    Component* RUArm = new Component("R_UpperArm");
    RUArm->jointTranslate = vec3(0.6f, 0.7f, 0); RUArm->meshTranslate = vec3(0.4f, 0, 0); RUArm->meshScale = vec3(0.8f, 0.25f, 0.25f); RUArm->animate = true;
    robotRoot->children.push_back(RUArm);

    Component* RLArm = new Component("R_LowerArm");
    RLArm->jointTranslate = vec3(0.8f, 0, 0); RLArm->meshTranslate = vec3(0.35f, 0, 0); RLArm->meshScale = vec3(0.7f, 0.2f, 0.2f);
    RUArm->children.push_back(RLArm);

    Component* LULeg = new Component("L_UpperLeg");
    LULeg->jointTranslate = vec3(-0.3f, -0.9f, 0); LULeg->meshTranslate = vec3(0, -0.5f, 0); LULeg->meshScale = vec3(0.35f, 1.0f, 0.35f);
    robotRoot->children.push_back(LULeg);

    Component* LLLeg = new Component("L_LowerLeg");
    LLLeg->jointTranslate = vec3(0, -1.0f, 0); LLLeg->meshTranslate = vec3(0, -0.4f, 0); LLLeg->meshScale = vec3(0.25f, 0.8f, 0.25f);
    LULeg->children.push_back(LLLeg);

    Component* RULeg = new Component("R_UpperLeg");
    RULeg->jointTranslate = vec3(0.3f, -0.9f, 0); RULeg->meshTranslate = vec3(0, -0.5f, 0); RULeg->meshScale = vec3(0.35f, 1.0f, 0.35f);
    robotRoot->children.push_back(RULeg);

    Component* RLLeg = new Component("R_LowerLeg");
    RLLeg->jointTranslate = vec3(0, -1.0f, 0); RLLeg->meshTranslate = vec3(0, -0.4f, 0); RLLeg->meshScale = vec3(0.25f, 0.8f, 0.25f);
    RULeg->children.push_back(RLLeg);

    robotRoot->flatten(componentList);
    selectedComponent = componentList[0];
}

int main(int argc, char** argv) {
    if (argc < 2) return 0;
    RESOURCE_DIR = argv[1] + string("/");
    if (!glfwInit()) return -1;
    window = glfwCreateWindow(1024, 768, "Avanthika", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glewExperimental = true;
    if (glewInit() != GLEW_OK) return -1;

    glfwSetCharCallback(window, char_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    init();

    while (!glfwWindowShouldClose(window)) {
        double t = glfwGetTime();

        // Apply spin animation
        for (auto c : componentList) {
            if (c->animate && (c->meshScale.x > c->meshScale.y)) {
                // uses constant
                c->animatedAngles.x = (float)t * SPIN_SPEED; 
            }
        }

        render();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}