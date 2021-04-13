#include<iostream>

#define GL_SILENCE_DEPRECATION

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

/** STRUCTS */
struct Camera {
	glm::vec3 position;
	float yaw, pitch;
  float fovy, aspect, near, far;

	glm::mat4 get_view_projection() const {
    
    glm::mat4 rotate = get_rotation();

    glm::mat4 translate = glm::translate(glm::identity<glm::mat4>(), position);

    glm::mat4 projection = glm::perspective(this->fovy, this->aspect, this->near, this->far);
		return projection * glm::inverse(translate * rotate);
	}

  glm::mat4 get_rotation() const {
     glm::mat4 rotate = glm::rotate(glm::identity<glm::mat4>(), 
      glm::radians<float>(this->yaw), glm::vec3(0.0, 1.0, 0.0)
    );
    return glm::rotate(rotate, glm::radians<float>(this->pitch), glm::vec3(1.0, 0.0, 0.0));
  }
};

struct Clock {
public:
	Clock() : current_tick(glfwGetTime ()), last_tick(current_tick) {}

	// Returns the time in seconds since last call to `tick`.
	float tick () {
		this->last_tick    = this->current_tick;
		this->current_tick = glfwGetTime();

		return this->current_tick - this->last_tick;
	}

private:
	double current_tick;
	double last_tick;
};

struct Window {
public:
  Window() {
    if (!glfwInit()) {
      std::cout << "Error: Could not initialize glfw" << std::endl;
      exit(EXIT_FAILURE);
    }
  
    glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL 


    window = glfwCreateWindow(1260, 1080, "Water rendering", NULL, NULL);
    if (!window) {
      glfwTerminate();
      std::cout << "Could not create glfw window" << std::endl;
      exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);

    // Necessary glew initialization done here, because why not?
    glewExperimental = GL_TRUE; 
    GLenum err = glewInit();
    if (GLEW_OK != err) {
      /* Problem: glewInit failed, something is seriously wrong. */
      fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    }
  
    std::cout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;
    std::cout << "Using GL version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "Shading language version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl; 
  }

  ~Window() {
    glfwTerminate();
  }

  bool should_close () {
    return glfwWindowShouldClose(window);
  }

  void poll_events () {
    glfwPollEvents();
  }

  void swap_buffers () {
    glfwSwapBuffers(window);
  }

  bool is_key_pressed(int keycode) const {
    return glfwGetKey(window, keycode) == GLFW_PRESS;
  }

private:
  GLFWwindow* window;
};

/** FUNCTIONS */
void update(const Window& window, double dt, Camera& camera);
GLuint create_shader_program();
GLuint create_model_vao(float* data, size_t data_size, uint32_t* indices, size_t indices_size);

const char* vertex_shader_code = R"(
#version 410 core
layout(location = 0) in vec3 a_Pos;

uniform mat4 u_ViewProjection;
void main()
{
    gl_Position = u_ViewProjection * vec4(a_Pos, 1.0); 
}
)";

const char* fragment_shader_code = R"(
#version 410 core
out vec4 color;

void main()
{
    color = vec4(1.0, 0.0, 0.0, 1.0);
}
)";

int main(void)
{
  Window window;
  Clock clock;

  GLuint shader_program = create_shader_program();
  glUseProgram(shader_program);

  float cube_data[] = {
    -0.5f, -0.5f, -0.5f, // 0
    0.5f, -0.5f, -0.5f, // 1
    -0.5f, 0.5f, -0.5f, // 2
    0.5f, 0.5f, -0.5f, // 3
    -0.5f, -0.5f, 0.5f, // 4
    0.5f, -0.5f, 0.5f, // 5
    -0.5f, 0.5f, 0.5f, // 6
    0.5f, 0.5f, 0.5f, // 7
  };

  uint32_t cube_indices[] = {
    1, 0, 3, // Back
    3, 0, 2, 
    2, 0, 4, // Left
    4, 6, 2,
    4, 5, 7, // Front
    7, 6, 4,
    3, 7, 5, // Right
    5, 1, 3,
    2, 6, 7, // Top
    7, 3, 2,
    4, 0, 1, // Bottom
    1, 5, 4,
  };
  GLuint cube_vao = create_model_vao(cube_data, sizeof(cube_data), cube_indices, sizeof(cube_indices));

  Camera camera = { .position = glm::vec3(0.0, 2.0, 5.0),
	  .yaw = 0.0, .pitch = -20.0,
    .fovy = 45.0f, 1260.0f / 1080.0f, 0.01, 1000.0
  };

  while (!window.should_close ())
  {
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0.8, 0.85, 1.0, 1.0);

    update(window, clock.tick(), camera);
    
    glBindVertexArray(cube_vao);
    glm::mat4 view_projection = camera.get_view_projection();
    GLuint view_proj_loc = glGetUniformLocation(shader_program, "u_ViewProjection");
    glUniformMatrix4fv(view_proj_loc, 1, false, &view_projection[0][0]);
    glDrawElements(GL_TRIANGLES, sizeof(cube_indices) / 3.0f, GL_UNSIGNED_INT, 0);

    window.swap_buffers ();
    window.poll_events ();
  }
  return 0;
}

void update(const Window& window, double dt, Camera& camera) {
  // std::cout << "Frame rate: " << 1.0 / dt << " FPS" << std::endl;
  if (window.is_key_pressed(GLFW_KEY_LEFT)) {
    camera.yaw += 100 * dt;
  }
  if (window.is_key_pressed(GLFW_KEY_RIGHT)) {
    camera.yaw -= 100 * dt;
  }
  if (window.is_key_pressed(GLFW_KEY_UP)) {
    camera.pitch += 100 * dt;
  }
  if (window.is_key_pressed(GLFW_KEY_DOWN)) {
    camera.pitch -= 100 * dt;
  }

  glm::vec3 forward = glm::vec3(camera.get_rotation() * glm::vec4(0.0, 0.0, -1.0, 0.0));
  glm::vec3 right = glm::vec3(camera.get_rotation() * glm::vec4(1.0, 0.0, 0.0, 0.0));

  if (window.is_key_pressed(GLFW_KEY_W)) {
    camera.position += forward * (float) dt;
  }
  if (window.is_key_pressed(GLFW_KEY_S)) {
    camera.position -= forward * (float) dt;
  }
  if (window.is_key_pressed(GLFW_KEY_D)) {
    camera.position += right * (float) dt;
  }
  if (window.is_key_pressed(GLFW_KEY_A)) {
    camera.position -= right * (float) dt;
  }
}

GLuint create_shader_program() {
  int success;
  char infoLog[512];

  GLuint vs = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vs, 1, &vertex_shader_code, NULL);
  glCompileShader(vs);
  glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
  if(!success) {
    glGetShaderInfoLog(vs, sizeof(infoLog), NULL, infoLog);
    std::cout << "ERROR: Vertex shader compilation failed\n" << infoLog << std::endl;
  };

  GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fs, 1, &fragment_shader_code, NULL);
  glCompileShader(fs);
  glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
  if(!success) {
    glGetShaderInfoLog(fs, sizeof(infoLog), NULL, infoLog);
    std::cout << "ERROR: Fragment shader compilation failed\n" << infoLog << std::endl;
  };

  GLuint program = glCreateProgram();
  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if(!success) {
    glGetProgramInfoLog(program, sizeof(infoLog), NULL, infoLog);
    std::cout << "ERROR: Shader program linkage failed\n" << infoLog << std::endl;
  }  
  glDeleteShader(vs);
  glDeleteShader(fs);
  return program;
};

GLuint create_model_vao(float* data, size_t data_size, uint32_t* indices, size_t indices_size) {
  GLuint vao;
  glGenVertexArrays(1, &vao);

  GLuint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, data_size, data, GL_STATIC_DRAW);

  GLuint ebo;
  glGenBuffers(1, &ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_size, indices, GL_STATIC_DRAW);

  glBindVertexArray(vao);
  
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

  return vao;
}