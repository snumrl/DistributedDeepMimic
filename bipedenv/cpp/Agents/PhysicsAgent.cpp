#include "Render/Shader.h"
#include "Render/GLFWMain.h"
#include "Agents/PhysicsAgent.h"
#include "Agents/DefaultAgent.h"
#include "Render/DART_interface.h"
#include "MotionDatabase/SkeletonBuilder.h"
#include "Helper/Functions.h"
#include "Parameter.h"
#include<vector>
#include<tuple>

WorldPtr PhysicsAgent::physicsWorld;

WorldPtr PhysicsAgent::getPhysicsWorld(){
    if(!physicsWorld){
		physicsWorld = World::create();

		const char* groundFile = "./character/ground.xml";
		printf("Ground File: %s\n", groundFile);
		SkeletonPtr ground = DPhy::SkeletonBuilder::BuildFromFile(groundFile);

		physicsWorld->addSkeleton(ground);

		printf("Skeleton File: %s\n", Parameter::humanoidFile.c_str());
		SkeletonPtr biped = DPhy::SkeletonBuilder::BuildFromFile(Parameter::humanoidFile);
		
		physicsWorld->addSkeleton(biped);
		physicsWorld->setGravity(Eigen::Vector3d(0.0, -9.81, 0.0));
		physicsWorld->setTimeStep(1.0/Parameter::physicsFPS);
		physicsWorld->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
		dynamic_cast<dart::constraint::BoxedLcpConstraintSolver*>(physicsWorld->getConstraintSolver())
			->setBoxedLcpSolver(std::make_shared<dart::constraint::PgsBoxedLcpSolver>());
	}
	return physicsWorld->clone();
}

PhysicsAgent::PhysicsAgent() : DefaultAgent(NULL, Eigen::Vector3d(0.5, 1, 0.5)), mWorld(getPhysicsWorld()){
	skel = mWorld->getSkeleton("Humanoid"); 
}

void PhysicsAgent::step(const Eigen::VectorXd &torque){
	Eigen::VectorXd totalTorque = torque;
	totalTorque.segment(0, 6) = Eigen::Vector6d::Zero();
	skel->setForces(totalTorque);
	mWorld->step();
}

void PhysicsAgent::reset(const Eigen::VectorXd &startPosition, const Eigen::VectorXd &startVelocity)
{
	skel->setPositions(startPosition);
	skel->setVelocities(startVelocity);
	skel->computeForwardKinematics(true, true, false);
}

Eigen::VectorXd PhysicsAgent::getPosition(){ return skel->getPositions(); }
Eigen::VectorXd PhysicsAgent::getVelocity(){ return skel->getVelocities(); }

void PhysicsAgent::display(){
	DefaultAgent::display();
}
