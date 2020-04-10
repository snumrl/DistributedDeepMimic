#include "Render/MyGLFWwindow.h"
#include "Render/GLFWMain.h"

static void initLights(double x = 0, double z = 0, double fx = 0, double fz = 0){
	static float ambient[]             = {0.02, 0.02, 0.02, 1.0};
	static float diffuse[]             = {.1, .1, .1, 1.0};

	//  static float ambient0[]            = {.01, .01, .01, 1.0};
	//  static float diffuse0[]            = {.2, .2, .2, 1.0};
	//  static float specular0[]           = {.1, .1, .1, 1.0};

	static float ambient0[]            = {.15, .15, .15, 1.0};
	static float diffuse0[]            = {.2, .2, .2, 1.0};
	static float specular0[]           = {.1, .1, .1, 1.0};


	static float spot_direction0[]     = {0.0, -1.0, 0.0};


	static float ambient1[]            = {.02, .02, .02, 1.0};
	static float diffuse1[]            = {.01, .01, .01, 1.0};
	static float specular1[]           = {.01, .01, .01, 1.0};

	static float ambient2[]            = {.01, .01, .01, 1.0};
	static float diffuse2[]            = {.17, .17, .17, 1.0};
	static float specular2[]           = {.1, .1, .1, 1.0};

	static float ambient3[]            = {.06, .06, .06, 1.0};
	static float diffuse3[]            = {.15, .15, .15, 1.0};
	static float specular3[]           = {.1, .1, .1, 1.0};


	static float front_mat_shininess[] = {24.0};
	static float front_mat_specular[]  = {0.2, 0.2,  0.2,  1.0};
	static float front_mat_diffuse[]   = {0.2, 0.2, 0.2, 1.0};
	static float lmodel_ambient[]      = {0.2, 0.2,  0.2,  1.0};
	static float lmodel_twoside[]      = {GL_TRUE};

	GLfloat position0[] = {0.0, 3.0, 0.0, 1.0};
	position0[0] = x;
	position0[2] = z;

	GLfloat position1[] = {0.0, 1.0, -1.0, 0.0};

	GLfloat position2[] = {0.0, 5.0, 0.0, 1.0};
	position2[0] = x;
	position2[2] = z;

	GLfloat position3[] = {0.0, 1.3, 0.0, 1.0};
	position3[0] = fx;
	position3[2] = fz;

	// glClear(GL_COLOR_BUFFER_BIT);

	// glEnable(GL_LIGHT0);
	// glLightfv(GL_LIGHT0, GL_AMBIENT,  ambient);
	// glLightfv(GL_LIGHT0, GL_DIFFUSE,  diffuse);
	// glLightfv(GL_LIGHT0, GL_POSITION, position);

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT,  lmodel_ambient);
	glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambient0);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse0);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specular0);
	glLightfv(GL_LIGHT0, GL_POSITION, position0);
	glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, spot_direction0);
	glLightf(GL_LIGHT0,  GL_SPOT_CUTOFF, 30.0);
	glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 2.0);
	glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 1.0);
	glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 1.0);

	glEnable(GL_LIGHT1);
	glLightfv(GL_LIGHT1, GL_AMBIENT, ambient1);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse1);
	glLightfv(GL_LIGHT1, GL_SPECULAR, specular1);
	glLightfv(GL_LIGHT1, GL_POSITION, position1);
	glLightf(GL_LIGHT1, GL_CONSTANT_ATTENUATION, 2.0);
	glLightf(GL_LIGHT1, GL_LINEAR_ATTENUATION, 1.0);
	glLightf(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, 1.0);


	glEnable(GL_LIGHT2);
	glLightfv(GL_LIGHT2, GL_AMBIENT, ambient2);
	glLightfv(GL_LIGHT2, GL_DIFFUSE, diffuse2);
	glLightfv(GL_LIGHT2, GL_SPECULAR, specular2);
	glLightfv(GL_LIGHT2, GL_POSITION, position2);
	glLightf(GL_LIGHT2, GL_CONSTANT_ATTENUATION, 2.0);
	glLightf(GL_LIGHT2, GL_LINEAR_ATTENUATION, 1.0);
	glLightf(GL_LIGHT2, GL_QUADRATIC_ATTENUATION, 1.0);

	glEnable(GL_LIGHT3);
	glLightfv(GL_LIGHT3, GL_AMBIENT, ambient3);
	glLightfv(GL_LIGHT3, GL_DIFFUSE, diffuse3);
	glLightfv(GL_LIGHT3, GL_SPECULAR, specular3);
	glLightfv(GL_LIGHT3, GL_POSITION, position3);
	glLightf(GL_LIGHT3, GL_CONSTANT_ATTENUATION, 2.0);
	glLightf(GL_LIGHT3, GL_LINEAR_ATTENUATION, 1.0);
	glLightf(GL_LIGHT3, GL_QUADRATIC_ATTENUATION, 1.0);

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	// glColorMaterial(GL_FRONT_AND_BACK, GL_SPECULAR);

	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, front_mat_shininess);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  front_mat_specular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   front_mat_diffuse);

	// GLfloat fogColor[] = {1.0, 1.0, 1.0, 1.0};

	// glFogi(GL_FOG_MODE, GL_LINEAR);
	// glFogfv(GL_FOG_COLOR, fogColor);
	// glFogf(GL_FOG_DENSITY, 10);
	// glHint(GL_FOG_HINT, GL_DONT_CARE);
	// glFogf(GL_FOG_START, 0.);
	// glFogf(GL_FOG_END, 1.);
	// glEnable(GL_FOG);
}

MyGLFWwindow::MyGLFWwindow(int w, int h, const char* name, GLFWmonitor* monitor, GLFWwindow* share):
	w(w), h(h){
	GLFWMain::windowList.emplace_back(this);
	window = glfwCreateWindow(w, h, name, monitor, share);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(2);

	glfwSetMouseButtonCallback(window, GLFWMain::mouse_button_callback);
	glfwSetCursorPosCallback(window, GLFWMain::cursor_position_callback);
	glfwSetKeyCallback(window, GLFWMain::key_callback);
	
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glEnable(GL_NORMALIZE);
	glShadeModel(GL_SMOOTH);
	initLights();

	glClearColor(0.9, 0.9, 0.9, 1);
}

void MyGLFWwindow::addObject(DrawablePtr obj){
	obj->reshape(w, h); objects.push_back(obj);
}

void MyGLFWwindow::removeObject(DrawablePtr obj){
	for(auto it = objects.begin(); it != objects.end(); ++it){
		if(*it == obj){
			objects.erase(it);
			break;
		}
	}
}

void MyGLFWwindow::display(){ 
	int cw, ch; glfwGetWindowSize(window, &cw, &ch);
	if(cw != w || ch != h) reshape(cw, ch);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	for(auto obj : objects) obj->display();
}

void MyGLFWwindow::mouse_button_callback(GLFWwindow* window, int button, int action, int mode){
	for(auto obj : objects) obj->mouse_button_callback(window, button, action, mode);
}

void MyGLFWwindow::cursor_position_callback(GLFWwindow* window, double xpos, double ypos){
	for(auto obj : objects) obj->cursor_position_callback(window, xpos, ypos);
}

void MyGLFWwindow::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
	for(auto obj : objects) obj->key_callback(window, key, scancode, action, mods);
}

void MyGLFWwindow::reshape(int w, int h){
	this->w = w; this->h = h;
	for(auto obj : objects) obj->reshape(w, h);
}