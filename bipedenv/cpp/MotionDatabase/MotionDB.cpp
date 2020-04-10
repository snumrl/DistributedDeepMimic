#include "MotionDatabase/MotionDB.h"
#include "MotionDatabase/SkeletonBuilder.h"
#include "MotionDatabase/MotionHelper.h"
#include "Parameter.h"
#include "Helper/Functions.h"
#include<tinyxml2.h>

using namespace tinyxml2;

// Parsing

namespace BVHParsing{
	struct BVHNode{
		std::string name;
		double offset[3];
		std::vector<std::string> channelList;
		std::vector<BVHNode*> child;

		// WARNING: only BallJoint and FreeJoint can be accepted
		Eigen::VectorXd readFile(std::ifstream &file){
			if(channelList.size() == 0) return Eigen::VectorXd(0);
			Eigen::VectorXd result = Eigen::VectorXd::Zero(channelList.size());
			Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
			for(std::string channel : channelList){
				double value;
				file >> value;
				if(channel.substr(1) == "rotation"){
					Eigen::Vector3d axis;
					switch(channel[0]){
						case 'X': axis = Eigen::Vector3d::UnitX(); break;
						case 'Y': axis = Eigen::Vector3d::UnitY(); break;
						case 'Z': axis = Eigen::Vector3d::UnitZ(); break;
						default: printf("BVH Parsing Error: Can't understand \"%s\"", channel.c_str());
					}
					rotation = rotation * Eigen::AngleAxisd(value * PI / 180, axis);
				}
				else if(channel.substr(1) == "position"){
					int idx;
					switch(channel[0]){
						case 'X': idx = 3; break;
						case 'Y': idx = 4; break;
						case 'Z': idx = 5; break;
						default: printf("BVH Parsing Error: Can't understand \"%s\"", channel.c_str()); exit(-1);
					}
					result[idx] = value / 100.0;
				}
			}
			result.segment(0, 3) = dart::dynamics::BallJoint::convertToPositions(rotation);
			return result;
		}
	};

	void getBVHToSkel(std::string file, std::map<std::string, int> &bvhToSkel){
		XMLDocument doc; doc.LoadFile(file.c_str());
		XMLElement *skeldoc = doc.FirstChildElement("Skeleton");

		int cnt = 0;
		for(XMLElement *body = skeldoc->FirstChildElement("Joint"); body != nullptr; body = body->NextSiblingElement("Joint")){
			bvhToSkel[body->Attribute("bvh")] = cnt;
			std::string jointType = body->Attribute("type");
			if(jointType == "FreeJoint") cnt += 6;
			else cnt += 3;
		}
	}

	BVHNode* endParser(std::ifstream &file, std::vector<BVHNode*> &nodeList){
		BVHNode* current_node = new BVHNode();
		nodeList.push_back(current_node);
		std::string tmp;
		file >> current_node->name;
		file >> tmp; assert(tmp == "{");
		file >> tmp; assert(tmp == "OFFSET");
		for(int i = 0; i < 3; i++) file >> current_node->offset[i];
		file >> tmp; assert(tmp == "}");

		return current_node;
	}

	BVHNode* nodeParser(std::ifstream &file, std::vector<BVHNode*> &nodeList)
	{
		std::string command, tmp;
		int sz;
		BVHNode* current_node = new BVHNode();
		nodeList.push_back(current_node);

		file >> current_node->name;
		file >> tmp; assert(tmp == "{");

		while(1){
			file >> command;
			if(command == "OFFSET"){
				for(int i = 0; i < 3; i++) file >> current_node->offset[i];
			}
			else if(command == "CHANNELS"){
				file >> sz;
				for(int i = 0; i < sz; i++){
					file >> tmp;
					current_node->channelList.push_back(tmp);
				}
			}
			else if(command == "JOINT"){
				current_node->child.push_back(nodeParser(file, nodeList));
			}
			else if(command == "End"){
				current_node->child.push_back(endParser(file, nodeList));
			}
			else if(command == "}") break;
		}
		return current_node;
	}

	void motionParser(std::ifstream &file, int channels, std::vector<Eigen::VectorXd> &motion, std::vector<BVHNode*> &nodeList, std::map<std::string, int> &bvhToSkel)
	{
		double time_interval;
		int frames;
		std::string command;

		file >> command; assert(command == "Frames:");
		file >> frames;
		file >> command; assert(command == "Frame");
		file >> command; assert(command == "Time:");
		file >> time_interval; 
//		fps = 1. / time_interval; printf("fps: %lf\n", fps);

		for(int t = 0; t < frames; t++){
			// Position
			Eigen::VectorXd vec = Eigen::VectorXd::Zero(channels);
			for(auto node : nodeList){
				Eigen::VectorXd tmp = node->readFile(file);
				if(tmp.size() == 0) continue;
				if(bvhToSkel.find(node->name) == bvhToSkel.end()) {
					printf("(bvh)%s is not a skeleton joint\n", node->name.c_str());
					motion.clear();
					return;
				}
				vec.segment(bvhToSkel[node->name], tmp.size()) = tmp;
			}
			motion.push_back(vec);
		}
	}
	std::vector<Eigen::VectorXd> Parser(std::string filename, std::map<std::string, int> &bvhToSkel, int channels){
		std::ifstream file(filename);
		std::string command, tmp;
		std::vector<Eigen::VectorXd> motion;
		std::vector<BVHNode*> nodeList;

		while(1){
			command = ""; file >> command;
			if(command == "HIERARCHY"){
				file >> tmp; assert(tmp == "ROOT");
				/* BVHNode* root = */ nodeParser(file, nodeList);
			}
			else if(command == "MOTION") motionParser(file, channels, motion, nodeList, bvhToSkel);
			else break;
		}
		file.close();
		return motion;
	}
};

MotionDB::MotionDB(std::vector<std::string> files, SkeletonPtr skel) : skel(skel){
	assert(skel->getName() == "Humanoid");
	motionData.reserve(files.size());
	footContactData.reserve(files.size());
	std::map<std::string, int> bvhToSkel;
	BVHParsing::getBVHToSkel(Parameter::humanoidFile, bvhToSkel);
	for(std::string file : files){
		std::vector<Eigen::VectorXd> data;
		std::vector<int> foot;
		data = BVHParsing::Parser(file, bvhToSkel, (int)skel->getNumDofs());
		if(data.size() == 0){
			printf("Error on BVH parsing\n");
			continue;
		}
		else printf("%s accepted\n", file.c_str());
		int clip = motionData.size();
		motionData.push_back(data);

		for(int i = 0; i+1 < (int)data.size(); i++){
			skel->setPositions(getPosition(clip, i));
			skel->setVelocities(getVelocity(clip, i));
			skel->computeForwardKinematics(true, true, false);
			
			int value = 0;
			for(int j = 0; j < (int)Parameter::footList.size(); j++)
				value |= ::isFootContact(skel->getBodyNode(Parameter::footList[j])) << j;
			foot.push_back(value);
		}
		footContactData.push_back(foot);
	}
}

int MotionDB::numClip(){ return motionData.size(); }
int MotionDB::numFrame(int frame){ return motionData[frame].size() - 1; }

int MotionDB::footContactInfo(int clip, int frame){
	if((int)footContactData[clip].size() <= frame){
		printf("WARNING(foot contact): motiondata size(%d) <= status.frame(%d) + 1\n", (int)footContactData[clip].size(), frame);
		int sz = footContactData[clip].size();
		return footContactData[clip][frame % sz];
	}
	return footContactData[clip][frame];
}

int MotionDB::footContactInfo(int clip, double frame){
	if((int)footContactData[clip].size() <= frame){
		printf("WARNING(foot contact): motiondata size(%d) <= status.frame(%d) + 1\n", (int)footContactData[clip].size(), frame);
		int sz = footContactData[clip].size();
		return footContactInfo(clip, fmod(frame, sz));
	}
	return footContactData[clip][(int)(frame + 0.5)];
}

bool MotionDB::isFootContact(int clip, int frame, int foot){
	return footContactInfo(clip, frame) & 1<<foot;
}

Eigen::VectorXd MotionDB::getPosition(int clip, int frame){
	if((int)motionData[clip].size() <= frame+1){
		printf("WARNING(get position): motiondata size(%d) <= status.frame(%d) + 1\n", (int)motionData[clip].size(), frame);
		int sz = motionData[clip].size() - 1;
		return motionData[clip][frame % sz + 1];
	}
	return motionData[clip][frame+1];
}

Eigen::VectorXd MotionDB::getPosition(int clip, double frame){
	if((int)motionData[clip].size() <= frame+1){
		printf("WARNING(get position): motiondata size(%d) <= status.frame(%d) + 1\n", (int)motionData[clip].size(), frame);
		int sz = motionData[clip].size() - 1;
		return getPosition(clip, fmod(frame, sz));
	}
	int t = (int)(frame + 1e-9);
	if(fabs(t-frame) < 1e-9) return getPosition(clip, t);
	return blendingPosition(getPosition(clip, t), getPosition(clip, t+1), frame-t, 1);
}

Eigen::VectorXd MotionDB::getVelocity(int clip, int frame){
	if((int)motionData[clip].size() <= frame+1){
		printf("WARNING: motiondata size <= status.frame + 1\n");
		int sz = motionData[clip].size() - 1;
		return skel->getPositionDifferences(motionData[clip][frame%sz + 1],motionData[clip][frame%sz]) * Parameter::kinematicsFPS;
	}
	return skel->getPositionDifferences(motionData[clip][frame+1], motionData[clip][frame]) * Parameter::kinematicsFPS;
}

Eigen::VectorXd MotionDB::getVelocity(int clip, double frame){
	if((int)motionData[clip].size() <= frame+1){
		printf("WARNING(get position): motiondata size(%d) <= status.frame(%d) + 1\n", (int)motionData[clip].size(), frame);
		int sz = motionData[clip].size() - 1;
		return getPosition(clip, fmod(frame, sz));
	}
	int t = (int)(frame + 1e-9);
	if(fabs(t-frame) < 1e-9) return getVelocity(clip, t);
	return blendingPosition(getVelocity(clip, t), getVelocity(clip, t+1), frame-t, 1);
}

void MotionDB::reshape(int width, int height){
	this->width = width; this->height = height;
}

static void set2DCamera(double width, double height){
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(100.0, (GLfloat)width / (GLfloat)height, 1e-6, 1000.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0, 0, 1,
			0, 0, 0,
			0, 1, 0);
}

void MotionDB::display()
{
	set2DCamera(width, height);
	glLineWidth(1.0);

	glColor3d(0.0, 0.0, 0.0);
	glNormal3d(0, 0, 1);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	int nclip = motionData.size();
	double W = 1.0 / nclip, S = -1, E = 1;
	glBegin(GL_LINES);
	for(int i = 0; i < nclip; i++){
		glVertex3d(W*(i - nclip/2), S, 0);
		glVertex3d(W*(i - nclip/2), E, 0);
	}
	glEnd();

	for(auto c : visitList){
		auto drawSquare = [](double x, double y, double len){
			glBegin(GL_POLYGON);
			glVertex3d(x + len, y, 0);
			glVertex3d(x, y + len, 0);
			glVertex3d(x - len, y, 0);
			glVertex3d(x, y - len, 0);
			glEnd();
		};
		double x = W*(c.first - nclip/2), y = (double)c.second / motionData[c.first].size() * (E-S) + S;
		drawSquare(x, y, 0.01);
	}
	visitList.clear();
}
