#include<tuple>
#include "Render/GLfunctions.h"
#include "Environment/DeepMimic.h"
#include "Environment/EnvironmentHelper.h"
#include "Helper/Functions.h"
#include "Parameter.h"
#include "Render/Shader.h"

const int KINEMATICS_FLAG = 1;
const int PHYSICS_FLAG = 2;
const int PHASE_FLAG = 4;

int DeepMimic::dof;

DeepMimic::DeepMimic(KinematicsPlayerPtr<int> kinematics, PhysicsSPDAgentPtr physics, Camera3DPtr camera) : 
	DefaultEnvironment(camera), kinematics(kinematics), physics(physics), displayStatus(PHYSICS_FLAG | PHASE_FLAG){
}

DeepMimic::DeepMimic(MotionDBPtr motionDB, Camera3DPtr camera) : 
	DeepMimic(KinematicsPlayerPtr<int>(new KinematicsPlayer<int>(motionDB)), PhysicsSPDAgentPtr(new PhysicsSPDAgent()), camera){
}

void DeepMimic::reset(){
	kinematics->reset();
	physics->reset(kinematics->getPosition(), kinematics->getVelocity());
	kinematics->step();
}

void DeepMimic::step(const Eigen::VectorXd &action, double &reward, int &done){
	Eigen::VectorXd clipedAction = Eigen::VectorXd::Zero(actionSize() + 6);
	clipedAction.segment(6, actionSize()) = (action * 0.1).cwiseMin(PI * 0.7).cwiseMax(-PI * 0.7);
	Eigen::VectorXd position = kinematics->getPosition() + clipedAction;

	int rep = 1;
	for(int i = 0; i < Parameter::physicsFPS / Parameter::kinematicsFPS / rep; i++){
		physics->step(position, kinematics->getVelocity(), rep);
	}
	std::tie(reward, done) = std::tuple<double, int>(getReward(), isTerminate());
	kinematics->step();
}

Eigen::VectorXd DeepMimic::getState(){
	Eigen::VectorXd state(observationSize());
	double phase = kinematics->getPhase();
	state << physics->getPosition(), physics->getVelocity(), cos(phase * 2 * PI), sin(phase * 2 * PI);
	
	state[3] -= kinematics->getPosition()[3];
	state[5] -= kinematics->getPosition()[5];
	return state;
}

double DeepMimic::getReward(){
	return getDeepMimicReward(physics->skel, kinematics->skel);
}

bool DeepMimic::isTerminate(){
	if(getDeepMimicEarlyTerminate(physics->skel, kinematics->skel)) return true;
	else return false;
}

void DeepMimic::display(){
	double x = kinematics->getPosition()[3], z = kinematics->getPosition()[5];
	glPushMatrix();
	glTranslated(1.0, 0.0, 1.0);
	alignCamera();
	if(isFlagSet(displayStatus, KINEMATICS_FLAG)) kinematics->display();
	if(isFlagSet(displayStatus, PHYSICS_FLAG)) physics->display();
	if(isFlagSet(displayStatus, PHASE_FLAG)) kinematics->displayPhase(camera);
	Shader::useShader(true);
	GUI::DrawGround(x, z, 0.0);
	Shader::useShader(false);
	glPopMatrix();
}

void DeepMimic::alignCamera(){
	if(camera){
		//Eigen::Vector3d dx = physics->skel->getPositions().segment(3, 3) - camera->origin;
		Eigen::Vector3d dx = physics->getPosition().segment(3, 3) - camera->origin;
		dx[1] = 0;
		camera->origin += dx; camera->camera += dx;
	}
}

void DeepMimic::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
	switch(action){
		case GLFW_PRESS: switch(key){
			case GLFW_KEY_A: flipFlag(displayStatus, KINEMATICS_FLAG); break;
			case GLFW_KEY_S: flipFlag(displayStatus, PHYSICS_FLAG); break;
		}
	}
}

int DeepMimic::observationSize(){
	return dof * 2 + 2;
}

int DeepMimic::getObservationSize(){
	return observationSize();
}

int DeepMimic::actionSize(){
	return dof - 6;
}

int DeepMimic::getActionSize(){
	return actionSize();
}
