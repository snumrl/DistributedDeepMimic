#ifndef AGENTHELPER
#define AGENTHELPER

#include<dart/dart.hpp>
using namespace dart::dynamics;

const int ROOT_HEIGHT = 1;
const int ROOT_DIFF = 2;
const int ROOT_ANGLE_DIFF = 4;

double getDeepMimicReward(SkeletonPtr physicsSkel, SkeletonPtr kinematicsSkel);
bool getDeepMimicEarlyTerminate(SkeletonPtr physicsSkel, SkeletonPtr kinematicsSkel, 
	int flag = ROOT_HEIGHT | ROOT_DIFF | ROOT_ANGLE_DIFF);

#endif