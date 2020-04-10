#include <boost/python/numpy.hpp>
#include "Helper/Functions.h"
#include "Parameter.h"
#include "Render/Render.h"
#include "Agents/Agents.h"
#include "Environment/Environment.h"
#include "MotionDatabase/SkeletonBuilder.h"
#include <iostream>
#include <Python.h>
#include <time.h>
//#include <omp.h>

#include "Helper/TimeProfiler.h"
#include "api.h"

using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart::math;
namespace p = boost::python;
namespace np = boost::python::numpy;

/* Render */
static MyGLFWwindow* w1, *w2;
static Camera3DPtr camera;
static SkeletonPtr baseSkel;
static MotionDBPtr motionDB;
static int dof;

/* Init serise */
void globalInit()
{
//	Eigen::initParallel();
//	omp_set_num_threads(16);

	Parameter::loadParameter();
	baseSkel = DPhy::SkeletonBuilder::BuildFromFile(Parameter::humanoidFile);
	motionDB = MotionDBPtr(new MotionDB(Parameter::motionDataFiles, baseSkel));
	dof = DeepMimic::dof = baseSkel->getNumDofs();
}

static void RenderInit()
{
	static bool init = false;
	if(init) return;
	init = true;

	w1 = GLFWMain::globalInit();
	camera = Camera3DPtr(new Camera3D(
		Eigen::Vector3d(0, 0, 0),
		Eigen::Vector3d(3, 3, 3),
		Eigen::Vector3d(0, 1, 0), 100.0));
	
	w1->addObject((DrawablePtr)camera);
}

/* python helper */
template<typename T, typename U>
static Eigen::Matrix<T, -1, -1> npToEigen2d(np::ndarray np){
	int r = np.shape(0), c = np.shape(1);
	Eigen::Matrix<T, -1, -1> mat(r, c);
	U* data = reinterpret_cast<U*>(np.get_data());
	for(int i = 0; i < r; i++)
		for(int j = 0; j < c; j++)
			mat(i, j) = data[i*c + j];
	return mat;
}

template<typename T, typename U>
static np::ndarray eigenTonp2d(const Eigen::Matrix<U, -1, -1> &mat){
	int r = mat.rows(), c = mat.cols();
	p::tuple shape = p::make_tuple(r, c);
	np::dtype dtype = np::dtype::get_builtin<T>();
	np::ndarray np = np::empty(shape, dtype);
	for(int i = 0; i < r; i++)
		for(int j = 0; j < c; j++)
			np[i][j] = mat(i, j);
	return np; 
}

template<typename T, typename U>
static Eigen::Matrix<T, -1, 1> npToEigen(np::ndarray np){
	int sz = np.shape(0); Eigen::Matrix<T, -1, 1> vec(sz);
	U* data = reinterpret_cast<U*>(np.get_data());
	for(int i = 0; i < sz; i++) vec[i] = data[i];
	return vec;
}

template<typename T, typename U>
static np::ndarray eigenTonp(const Eigen::Matrix<U, -1, 1> &vec){
	p::tuple shape = p::make_tuple(vec.size());
	np::dtype dtype = np::dtype::get_builtin<T>();
	np::ndarray np = np::empty(shape, dtype);
	for(int i = 0; i < vec.size(); i++) np[i] = vec[i];
	return np; 
}

/* render api */
void BipedEnv::globalRender(){
	GLFWMain::globalRender();
}

/* General environment Helper */
static std::vector<DefaultEnvironment*> envList;

static int envInit(DefaultEnvironment* env){
	envList.push_back(env);
	return envList.size() - 1;
}

/* General environment */
template<typename T>
int BipedEnv::GeneralEnvironmentInit(bool render){
	T* agent;
	if (render){
		RenderInit();
		agent = new T(motionDB, camera);
		w1->addObject((DrawablePtr) agent);
	}
	else agent = new T(motionDB);
	return envInit(agent);
}

Eigen::VectorXd BipedEnv::reset(int idx){
	DefaultEnvironment *agent = envList[idx];
	agent->reset();
	return agent->getState();
}

np::ndarray reset_python(int idx){
	return eigenTonp<float>(BipedEnv::reset(idx));
}

void BipedEnv::step(int idx, const Eigen::VectorXd &action, Eigen::VectorXd &next_state, double &reward, int &done){
	DefaultEnvironment *agent = envList[idx];
	agent->step(action, reward, done);
	if(done) agent->reset();
	agent->alignCamera();
	next_state = agent->getState();
}

boost::python::tuple step_python(int idx, np::ndarray action){
	double reward; int done;
	Eigen::VectorXd next_state;
	BipedEnv::step(idx, npToEigen<double, float>(action), next_state, reward, done);
	return p::make_tuple(eigenTonp<float>(next_state), reward, done, p::dict());
}

boost::python::tuple multistep_python(np::ndarray idx, np::ndarray npaction){
	Eigen::VectorXi id = npToEigen<int, int>(idx);
	Eigen::MatrixXd action = npToEigen2d<double, float>(npaction);
	int size = id.size(), os;
	if(size == 0) assert(false);
	os = envList[id[0]]->getObservationSize();
	assert(action.rows() == size && action.cols() == envList[id[0]]->getActionSize());

	Eigen::MatrixXd state(size, os);
	Eigen::VectorXd reward(size);
	Eigen::VectorXi done(size);

//#pragma omp parallel for schedule(dynamic)
	for(int i = 0; i < size; i++){
		DefaultEnvironment *agent = envList[id[i]];
		agent->step(action.row(i), reward[i], done[i]);
	}
	for(int i = 0; i < size; i++){
		DefaultEnvironment *agent = envList[id[i]];
		if(done[i]) agent->reset();
		state.row(i) = agent->getState();
	}
	envList[0]->alignCamera();
	return p::make_tuple(eigenTonp2d<float>(state), eigenTonp<float>(reward), \
		eigenTonp<int>(done), p::dict());
}

int BipedEnv::getObservationSize(int env){ return envList[env]->getObservationSize(); }
int BipedEnv::getActionSize(int env){ return envList[env]->getActionSize(); }

int GetSkeletonDOF(){ return dof; }

BOOST_PYTHON_MODULE(libbiped)
{
	globalInit();
	np::initialize();

	p::def("globalRender", BipedEnv::globalRender);

	p::def("step", step_python);
	p::def("multistep", multistep_python);
	p::def("reset", reset_python);

#define PYTHONENV(NAME) \
	p::def(#NAME "Init", BipedEnv::GeneralEnvironmentInit<NAME>);\
	p::def(#NAME "ObservationSize", NAME::observationSize);\
	p::def(#NAME "ActionSize", NAME::actionSize);\

DEFAULTENVIRONMENTLIST(PYTHONENV)
	
	p::def("GetSkeletonDOF", GetSkeletonDOF);
}

int main(){
	globalInit();
	int t = BipedEnv::GeneralEnvironmentInit<DeepMimic>(false);
	Eigen::VectorXd state = BipedEnv::reset(t);
	for(;;){
		Eigen::VectorXd action = Eigen::VectorXd::Zero(DeepMimic::actionSize());
		Eigen::VectorXd next_state;
		double reward; int done;
		BipedEnv::step(t, action, next_state, reward, done);
		if(done) state = BipedEnv::reset(t);
		else state = next_state;
	}
	printf("%lf %d\n", 0.0 / 0.0, std::isnan(0.0/0.0));
}
