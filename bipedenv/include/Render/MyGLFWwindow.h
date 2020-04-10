#ifndef MYGLFWWINDOW
#define MYGLFWWINDOW

#include "Drawable.h"
#include<GLFW/glfw3.h>
#include<vector>

class MyGLFWwindow{
public:
	MyGLFWwindow(int w, int h, const char* name, GLFWmonitor* monitor, GLFWwindow* share);
	GLFWwindow* window;

	void addObject(DrawablePtr obj);
	void removeObject(DrawablePtr obj);

	void display();
	
	void mouse_button_callback(GLFWwindow* window, int button, int action, int mode);
	void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
	void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
	void reshape(int w, int h);

	std::vector<DrawablePtr> objects;
	int w, h;
};

#endif