#include "Render/Shader.h"
#include "Render/DART_interface.h"
#include "Agents/DefaultAgent.h"
#include <dart/dart.hpp>

using namespace dart::dynamics;
using namespace dart::simulation;

DefaultAgent::DefaultAgent(SkeletonPtr skel, const Eigen::Vector3d &color) : skel(skel), color(color){}

void DefaultAgent::display(){
	Shader::useShader(true);
    GUI::DrawSkeleton(skel, color, 0);
	Shader::useShader(false);
}