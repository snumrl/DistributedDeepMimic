#ifndef GLFWMAIN
#define GLFWMAIN

#include "MyGLFWwindow.h"
#include <vector>
#include <set>
#include<Eigen/Dense>
#include <Eigen/Geometry>
#include <memory>

namespace GLFWMain{
	MyGLFWwindow* globalInit();
	void globalRender();
	extern std::vector<MyGLFWwindow*> windowList;
	extern std::set<int> joystickID;
	
	void display();
	void joystick(int jid, int event); 
	
	void mouse_button_callback(GLFWwindow* window, int button, int action, int mode);
	void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
	void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

	// helper
	MyGLFWwindow* getMyGLFWwindowPtr(GLFWwindow* window);
};

#endif
