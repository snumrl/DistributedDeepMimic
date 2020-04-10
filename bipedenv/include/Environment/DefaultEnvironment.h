#ifndef DEFAULTENVIRONMENT
#define DEFAULTENVIRONMENT

#include<Eigen/Dense>
#include<assert.h>
#include "Render/Camera3D.h"

class DefaultEnvironment{
public:
	DefaultEnvironment(Camera3DPtr camera);
	Camera3DPtr camera;

	virtual void reset() = 0;
	virtual void step(const Eigen::VectorXd &action, double &reward, int &done) = 0;
	virtual Eigen::VectorXd getState() = 0;

	virtual int getObservationSize() = 0;
	virtual int getActionSize() = 0;
	
	virtual void alignCamera() = 0;
};

#endif