#ifndef BIPEDENVAPI
#define BIPEDENVAPI

#include<vector>
#include<string>

// for general environment
#define DEFAULTENVIRONMENTLIST(MACRO) \
	MACRO(DeepMimic) \

#define ENVIRONMENTHEADER(NAME) class NAME;\

DEFAULTENVIRONMENTLIST(ENVIRONMENTHEADER);

namespace BipedEnv{
	void globalInit();
	void globalRender();
	template<typename T>
	int GeneralEnvironmentInit(bool render);
	int getObservationSize(int env);
	int getActionSize(int env);

	Eigen::VectorXd reset(int idx);
	void step(int idx, const Eigen::VectorXd &action, Eigen::VectorXd &next_state, double &reward, int &done);
};

#endif
