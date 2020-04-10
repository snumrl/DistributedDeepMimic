#ifndef KINEMATICSPLAYER
#define KINEMATICSPLAYER

#include "Agents/DefaultAgent.h"
#include "MotionDatabase/MotionStatus.h"
#include "Render/Drawable.h"
#include "Render/Camera3D.h"

using namespace dart::dynamics;

template<typename T>
class KinematicsPlayer : public DefaultAgent{
	public:
		KinematicsPlayer(MotionDBPtr motionDB);

		// For motion
		MotionDBPtr motionDB;
		MotionStatusPtr<T> currentStatus;

		// For action
		void setNextStatus(T time = 1);
		virtual void step(T time = 1);
		virtual void translate(double dx, double dz);
		virtual void reset();
		unsigned int seed_r;
		virtual void reset(MotionStatusPtr<T> currentStatus);
		virtual double getPhase();
		Eigen::VectorXd getPosition();
		Eigen::VectorXd getVelocity();
		MotionStatusPtr<T> getStatus();

		// For rendering
		virtual void display();
		virtual void displayPhase(Camera3DPtr camera);
};

template<typename T> using KinematicsPlayerPtr = std::shared_ptr<KinematicsPlayer<T>>;

#endif