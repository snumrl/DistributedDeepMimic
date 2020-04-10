#include "Render/Shader.h"

static const std::string vertexShader = "./bipedenv/Shader/vertexshader.txt";
static const std::string fragShader = "./bipedenv/Shader/fragshader.txt";

namespace Shader{
	GLuint program;
	void printShaderInfoLog(GLuint shader) {
		int len = 0;
		int charsWritten = 0;
		char* infoLog;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
		if (len > 0) {
			infoLog = (char*)malloc(len);
			glGetShaderInfoLog(shader, len, &charsWritten, infoLog);
			printf("%s\n", infoLog);
			free(infoLog);
		}
	}
	void printProgramInfoLog(GLuint obj) {
		int len = 0, charsWritten = 0;
		char* infoLog;
		glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &len);
		if (len > 0) {
			infoLog = (char*)malloc(len);
			glGetProgramInfoLog(obj, len, &charsWritten, infoLog);
			printf("%s\n", infoLog);
			free(infoLog);
		}
	}

	GLuint createProgram(std::string vertexShaderFile, std::string fragShaderFile){
		FILE *f = fopen(vertexShaderFile.c_str(), "rb");
		fseek(f, 0, SEEK_END);
		long fsize = ftell(f);
		fseek(f, 0, SEEK_SET);

		char *vert = (char *)malloc(fsize + 1);
		fread(vert, fsize, 1, f);
		fclose(f);

		vert[fsize] = 0;

		f = fopen(fragShaderFile.c_str(), "rb");
		fseek(f, 0, SEEK_END);
		fsize = ftell(f);
		fseek(f, 0, SEEK_SET);

		char *frag = (char *)malloc(fsize + 1);
		fread(frag, fsize, 1, f);
		fclose(f);

		frag[fsize] = 0;

		auto createShader = [](const char* src, GLenum type) {
			GLuint shader = glCreateShader(type);
			glShaderSource(shader, 1, &src, NULL);
			glCompileShader(shader);
			printShaderInfoLog(shader);
			return shader;
		};

		GLuint vertShader = createShader(vert, GL_VERTEX_SHADER);
		GLuint fragShader = createShader(frag, GL_FRAGMENT_SHADER);
		GLuint program = glCreateProgram();
		glAttachShader(program, vertShader);
		glAttachShader(program, fragShader);
		glLinkProgram(program);
		printProgramInfoLog(program);
		return program;
	}

	void useShader(int on){
		if(!program) program = createProgram(vertexShader, fragShader);
		glUseProgram(on ? program : 0);
	}
}
