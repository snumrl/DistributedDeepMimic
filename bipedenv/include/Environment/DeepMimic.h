#ifndef DEEPMIMIC
#define DEEPMIMIC

#include "Agents/KinematicsPlayer.h"
#include "Agents/PhysicsSPDAgent.h"
#include "Environment/DefaultEnvironment.h"
#include "Render/Drawable.h"
#include "Render/Camera3D.h"

class DeepMimic : public Drawable, public DefaultEnvironment{
public:
	DeepMimic(KinematicsPlayerPtr<int> kinematics, PhysicsSPDAgentPtr physics, Camera3DPtr camera = NULL);
	DeepMimic(MotionDBPtr motionDB, Camera3DPtr camera = NULL);
	KinematicsPlayerPtr<int> kinematics;
	PhysicsSPDAgentPtr physics;

	virtual void reset();
	virtual void step(const Eigen::VectorXd &action, double &reward, int &done);
	virtual Eigen::VectorXd getState();
	virtual int getObservationSize();
	virtual int getActionSize();

	int displayStatus;
	virtual void display();
	virtual void alignCamera();
	virtual void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

	static int dof;
	static int observationSize();
	static int actionSize();

private:
	virtual double getReward();
	virtual bool isTerminate();
};

#endif
