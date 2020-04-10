#include "Parameter.h"
#include "Render/GLfunctions.h"
#include "Render/GLFWMain.h"

#include <GLFW/glfw3.h>
#include<tuple>

// OpenGLSession & helper

std::vector<MyGLFWwindow*> GLFWMain::windowList;
std::set<int> GLFWMain::joystickID;

MyGLFWwindow* GLFWMain::globalInit(){
	char *fakeargv[] = { "fake", NULL };
	int fakeargc = 1;
	glutInit( &fakeargc, fakeargv );

	if (!glfwInit()) { 	 
		printf("GLFW init failed.\n");
		exit(EXIT_FAILURE);
	}
/*
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
//	*/
	glfwWindowHint(GLFW_SAMPLES, 4);

	MyGLFWwindow* f = new MyGLFWwindow(1600, 900, "World", nullptr, nullptr);

	GLenum err = glewInit();
	if (err != GLEW_OK) {
		printf("GLEW init failed.\n");
		printf("Error : %s\n", glewGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	glfwSetJoystickCallback(joystick);
	return f;
}

void GLFWMain::globalRender(){
	GLFWMain::display();
}

void GLFWMain::display(){
	static int counter = 0;
	static double last = 0, fps = 0;
	if(++counter == 20){
		double t = glfwGetTime();
		fps = 20.0 / std::max(0.05, t - last);
		last = t;
		counter = 0;
	}
	for(auto myWindow : windowList){
		if(glfwWindowShouldClose(myWindow->window)) exit(0);
		glfwMakeContextCurrent(myWindow->window);
		glfwPollEvents();
		myWindow->display();
		GUI::DrawStringOnScreen(-0.95, 0.95, "fps : " + std::to_string(fps), true, Eigen::Vector3d(1.0, 1.0, 0.0));
		glfwSwapBuffers(myWindow->window);
	}
}

void GLFWMain::joystick(int jid, int event){
	printf("joystick event, jid(%d), event(%d)\n", jid, event);
	if(event == 1) joystickID.insert(jid);
	else if(event == 0) joystickID.erase(jid);
}

MyGLFWwindow* GLFWMain::getMyGLFWwindowPtr(GLFWwindow* window){
	for(auto myWindow : windowList) if(myWindow->window == window) return myWindow;
	assert(false);
}

void GLFWMain::mouse_button_callback(GLFWwindow* window, int button, int action, int mode){
	getMyGLFWwindowPtr(window)->mouse_button_callback(window, button, action, mode);
}

void GLFWMain::cursor_position_callback(GLFWwindow* window, double xpos, double ypos){
	getMyGLFWwindowPtr(window)->cursor_position_callback(window, xpos, ypos);
}

void GLFWMain::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
	getMyGLFWwindowPtr(window)->key_callback(window, key, scancode, action, mods);
}
