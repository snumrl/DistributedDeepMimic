#include "Render/Shader.h"
#include "Render/GLfunctions.h"
#include "Agents/KinematicsPlayer.h"
#include "MotionDatabase/MotionHelper.h"
#include "MotionDatabase/SkeletonBuilder.h"
#include "Parameter.h"

template<typename T>
KinematicsPlayer<T>::KinematicsPlayer(MotionDBPtr motionDB) :
		DefaultAgent(DPhy::SkeletonBuilder::BuildFromFile(Parameter::humanoidFile), Eigen::Vector3d(0.0, 1.0, 0.0)),
		motionDB(motionDB), currentStatus(new MotionStatusLeaf<T>(motionDB, 0, rand()%motionDB->numFrame(0), 0, 0, 0)), seed_r(rand()){
	skel->setPositions(currentStatus->getPosition());
	skel->setVelocities(currentStatus->getVelocity());
	skel->computeForwardKinematics(true, true, false);
}

template<typename T>
void KinematicsPlayer<T>::setNextStatus(T time){
	if(currentStatus->isEnd(time)){
		time -= currentStatus->remainFrame();
		currentStatus = currentStatus->step(currentStatus, currentStatus->remainFrame());
		MotionStatusLeafPtr<T> to = MotionStatusLeafPtr<T>(new MotionStatusLeaf<T>(currentStatus->motionDB, 0, 0, 0, 0, 0));
		Eigen::VectorXd position = currentStatus->getPosition();
		setMotionAlignment(to, position.segment(0, 6));
		currentStatus = getMotionStatusOffset<T>(to, position, skel);
	}
	currentStatus = currentStatus->step(currentStatus, time);
}

template<typename T>
void KinematicsPlayer<T>::step(T time){
	setNextStatus(time);

	Eigen::VectorXd currentPosition = skel->getPositions();
	Eigen::VectorXd targetPosition = currentStatus->getPosition();
	Eigen::VectorXd targetVelocity = skel->getPositionDifferences(targetPosition, currentPosition) * (Parameter::kinematicsFPS / (double)time);
	skel->setPositions(targetPosition);
	skel->setVelocities(targetVelocity);
	skel->computeForwardKinematics(true, true, false);
}

template<typename T>
void KinematicsPlayer<T>::translate(double dx, double dz){
	currentStatus->translate(dx, dz);
}

template<typename T>
void KinematicsPlayer<T>::reset(){
	int frame = rand_r(&seed_r)%motionDB->numFrame(0);
	currentStatus = MotionStatusPtr<T>(new MotionStatusLeaf<T>(motionDB, 0, frame, 0, 0, 0));
	skel->setPositions(motionDB->getPosition(0, frame));
	skel->setVelocities(motionDB->getVelocity(0, frame));
	skel->computeForwardKinematics(true, true, false);
}

template<typename T>
void KinematicsPlayer<T>::reset(MotionStatusPtr<T> status){
	currentStatus = status->clone();
}

template<typename T>
double KinematicsPlayer<T>::getPhase(){
	return currentStatus->getPhase();
}

template<typename T>
Eigen::VectorXd KinematicsPlayer<T>::getPosition(){
	return skel->getPositions();
}

template<typename T>
Eigen::VectorXd KinematicsPlayer<T>::getVelocity(){
	return skel->getVelocities();
}

template<typename T>
void KinematicsPlayer<T>::display(){
	DefaultAgent::display();
}

template<typename T>
void KinematicsPlayer<T>::displayPhase(Camera3DPtr camera){
	
	const double PI = acos(-1);
	float currentColor[4]; glGetFloatv(GL_CURRENT_COLOR, currentColor);

	Shader::useShader(true);
	Eigen::Vector3d origin = Eigen::Vector3d(getPosition()[3], 2.0, getPosition()[5]);
	Eigen::Vector3d normal = camera->camera - camera->origin; normal[1] = 0; normal.normalize();
	Eigen::Vector3d up = Eigen::Vector3d::UnitY();
	double R = 0.2;
	GUI::Draw2DCircle(origin, normal, up, R);

	double a = 2*PI*getPhase();
	glColor3d(cos(a), sin(a), 1.0);
	GUI::DrawSphere(origin + normal.cross(up) * sin(a) * R + up * cos(a) * R, 0.05);

	Shader::useShader(false);
	glColor3fv(currentColor);
}

template class KinematicsPlayer<int>;
template class KinematicsPlayer<double>;