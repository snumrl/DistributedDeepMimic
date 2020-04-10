#include "Parameter.h"
#include <iostream>
#include<sstream>
#include<cmath>
#include<tinyxml2.h>

template<typename T>
void stringToData(const char* str, std::vector<T> &var){
	std::stringstream s(str);
	std::vector<T> res; T tmp;
	var.clear(); while(s >> tmp) var.push_back(tmp);
}

template<typename T>
void stringToData(const char* str, T &var){
	std::stringstream s(str);
	s >> var;
}

void Parameter::loadParameter(){
	tinyxml2::XMLDocument meta, param;
	if(meta.LoadFile("resources/metaData.xml")){
		std::cout << "Can't open file : " << "resources/metaData.xml" << std::endl;
		exit(-1);
	}
	auto tmp = meta.FirstChildElement("Parameter");
	if(!tmp){
		std::cout << "No parameter file spec" << std::endl;
		exit(-1);
	}
	if(param.LoadFile(tmp->GetText())){
		std::cout << "Can't open file : " << tmp->GetText() << std::endl;
		exit(-1);
	}
#define PARAMASSIGN(TYPE,NAME) if(auto elem = param.FirstChildElement(#NAME)) \
		stringToData(elem->GetText(), NAME);
	PARAM(PARAMASSIGN);
	
	maxFrame = featureFrame.empty() ? 60 : featureFrame.back();
	footContactLength = footList.size();
	poseLength = featureBodyVelocity.size() * 3 + featureBodyPosition.size() * 7;
	trajectoryLength = featureFrame.size() * 4;
	featureLength = footContactLength + poseLength + trajectoryLength;
}

namespace Parameter{
#define PARAMDECLARATION(TYPE,NAME) TYPE NAME;
	PARAM(PARAMDECLARATION)

	// motion matching helper function
	int maxFrame, featureLength;
	int poseLength, trajectoryLength, footContactLength;
}
