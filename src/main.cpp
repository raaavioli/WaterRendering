#include<iostream>
#include<vector>

#define GL_SILENCE_DEPRECATION

// CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cufft.h>

// GL
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Local
#include "wave_utils.h"

/** STRUCTS */
struct RawModel {
  RawModel(const std::vector<float>& data, const std::vector<uint32_t>& indices) {
    assert((indices.size() % 3) == 0);

    glGenVertexArrays(1, &this->renderer_id);
    glGenBuffers(1, &this->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), &data[0], GL_STATIC_DRAW);
    glGenBuffers(1, &this->ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint32_t), &indices[0], GL_STATIC_DRAW);
    this->index_count = indices.size();

    // Bind buffers to VAO
    bind();
    glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (const void *) (3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (const void *) (6 * sizeof(float)));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->ebo);
    unbind();
  };

  ~RawModel() {
    // Wish to delete buffers here but may cause problems
    /*glDeleteBuffers(1, ebo);
    glDeleteBuffers(1, vbo);
    glDeleteBuffers(1, renderer_id);*/
  }

  void bind() {
    glBindVertexArray(this->renderer_id);
  }

  void unbind() {
    glBindVertexArray(0);
  }

  void draw() {
    glDrawElements(GL_TRIANGLES, this->index_count, GL_UNSIGNED_INT, 0);
  }

private:
  GLuint renderer_id;
  GLuint vbo;
  GLuint ebo;

  uint32_t index_count;
};

struct Texture {
  Texture() {
    glGenTextures(1, &this->renderer_id);
    glBindTexture(GL_TEXTURE_2D, this->renderer_id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    char data[] = {(char) 255, (char) 255, (char) 255, (char) 255};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
  }

  Texture(const char* filename) {
    glGenTextures(1, &this->renderer_id);
    glBindTexture(GL_TEXTURE_2D, this->renderer_id);
    // set the texture wrapping/filtering options (on the currently bound texture object)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load and generate the texture
    int width, height, nrChannels;
    unsigned char *data = stbi_load(filename, &width, &height, &nrChannels, 0);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Error: Failed to load texture: " << filename << std::endl;
    }
    stbi_image_free(data);
  }

  void bind(uint32_t slot) {
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_2D, this->renderer_id);
  }

  void unbind() {
    glBindTexture(GL_TEXTURE_2D, 0);
  }

private:
  GLuint renderer_id;
};

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
	Clock() : current_tick(glfwGetTime ()), last_tick(current_tick), start_tick(current_tick) {}

	// Returns the time in seconds since last call to `tick`.
	float tick () {
		this->last_tick    = this->current_tick;
		this->current_tick = glfwGetTime();

		return this->current_tick - this->last_tick;
	}

  float since_start () {
    return glfwGetTime() - start_tick;
  }

private:
	double current_tick;
	double last_tick;
  const double start_tick;
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

    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
      /* Problem: glewInit failed, something is seriously wrong. */
      std::cerr << "Error: failed to initialize OpenGL context \n" << std::endl;
    }

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
RawModel create_cube();

static const char *_cudaGetErrorEnum(cufftResult error);

const char* vertex_shader_code = R"(
#version 410 core
layout(location = 0) in vec3 a_Pos;
layout(location = 1) in vec3 a_Color;
layout(location = 2) in vec2 a_UV;

layout(location = 0) out vec3 vs_Color;
layout(location = 1) out vec2 vs_UV;
layout(location = 2) out float vs_Time;

uniform mat4 u_ViewProjection;
uniform mat4 u_Model;
uniform float u_Time;

void main()
{
  vs_Time = u_Time;
  vs_Color = a_Color;
  vs_UV = a_UV;
  gl_Position = u_ViewProjection * u_Model * vec4(a_Pos, 1.0); 
}
)";

const char* fragment_shader_code = R"(
#version 410 core
out vec4 color;

layout(location = 0) in vec3 vs_Color;
layout(location = 1) in vec2 vs_UV;
layout(location = 2) in float vs_Time;

uniform sampler2D texture0;
uniform sampler2D texture1;

vec3 unwrap_normal(vec3 pixel) {
  return (pixel / 256.0f);
}

void main()
{
  float swing = sin(vs_Time) / 50.0f;
  vec3 tex0 = texture(texture0, vec2(vs_UV.x + vs_Time / 30.0f, vs_UV.y + vs_Time / 50.0f + swing)).xyz;
  vec3 tex1 = texture(texture1, vec2(vs_UV.x + vs_Time / 27.0f, vs_UV.y - vs_Time / 40.0f)).xyz;

  vec3 blended = normalize((unwrap_normal(tex0) + unwrap_normal(tex1)) / 2.0f);
  float shade = clamp(1 - pow(dot(blended, vec3(0.0, 0.0, 1.0)), 0.8), 0.22, 0.75);
  color = vec4(vs_Color * shade, 1.0);
}
)";

int main(void)
{
  Window window;
  Clock clock;

  GLuint shader_program = create_shader_program();
  glUseProgram(shader_program);

  RawModel cube = create_cube();

  std::vector<float> square_data = {
    -0.5, 0.0, -0.5, 0.6, 0.6, 0.9, 1.0, 1.0, 
    -0.5, 0.0, 0.5, 0.6, 0.6, 0.9, 1.0, 0.0,  
    0.5, 0.0, 0.5, 0.6, 0.6, 0.9, 0.0, 0.0,
    0.5, 0.0, -0.5, 0.6, 0.6, 0.9, 0.0, 1.0,

  };

  std::vector<uint32_t> square_indices = {
    3, 0, 1, 
    1, 2, 3
  };

  RawModel water(square_data, square_indices);
  //Texture water_texture("../water_image.jpg");
  Texture water_normal1("../NormalMap.png");
  Texture water_normal2("../NormalMap-2.png");
  Texture white_texture;
  Camera camera = { .position = glm::vec3(0.0, 4.0, 5.0),
	  .yaw = 0.0, .pitch = -20.0,
    .fovy = 45.0f, 1260.0f / 1080.0f, 0.01, 1000.0
  };

  glEnable(GL_CULL_FACE);

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS); 

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  GLuint view_proj_loc = glGetUniformLocation(shader_program, "u_ViewProjection");
  GLuint model_loc = glGetUniformLocation(shader_program, "u_Model");
  GLuint time_loc = glGetUniformLocation(shader_program, "u_Time");
  GLuint tex0_loc = glGetUniformLocation(shader_program, "texture0");
  GLuint tex1_loc  = glGetUniformLocation(shader_program, "texture1");

  glUniform1i(tex0_loc, 0);
  glUniform1i(tex1_loc, 1);

  // Wave simulation
  const int N = 512;
  double length = 1000;
  double two_pi = glm::two_pi<double>();

  std::complex<double>* h0_tk = new std::complex<double>[N * N]; // h0_tilde(k)
  std::complex<double>* h0_tmk = new std::complex<double>[N * N]; // h0_tilde(-k)

  for (int m = 0; m < N; m++) {
    for (int n = 0; n < N; n++) {
      int i = m * N + n;
      float kx = (n - N / 2.f) * two_pi / length;
      float kz = (m - N / 2.f) * two_pi / length;
      glm::vec2 k(kx, kz);
      h0_tk[i] = h0_tilde(k);
      h0_tmk[i] = h0_tilde(-k);
    }
  }

  std::cout << "Initializing CUDA context" << std::endl;
  cudaSetDevice(0);
  cudaFree(0);

  std::complex<double>* h_tk = new std::complex<double>[N * N]; // h_tilde(k, x, t)
  std::complex<double>* h_k = new std::complex<double>[N * N]; // h(k, x, t)

  cufftDoubleComplex* h_tk_device;
  cufftDoubleComplex* h_k_device;
  cudaError err = cudaMalloc ((void**) &h_tk_device, sizeof(std::complex<double>) * N * N);
  if (err != cudaSuccess)
    std::cout << "Error cudaMalloc" << err << std::endl;
  err = cudaMalloc ((void**) &h_k_device, sizeof(std::complex<double>) * N * N);
  if (err != cudaSuccess)
    std::cout << "Error cudaMalloc" << err << std::endl;
  std::cout << "Done allocating device buffers" << std::endl;

  // Setup h_tk + device
  for (int m = 0; m < N; m++) {
    for (int n = 0; n < N; n++) {
      int i = m * N + n;
      float kx = (n - N / 2.f) * two_pi / length;
      float kz = (m - N / 2.f) * two_pi / length;
      glm::vec2 K(kx, kz);
      h_tk[i] = h_tilde(h0_tk[i], h0_tmk[i], K, clock.since_start());
    }
  }
  err = cudaMemcpy(h_tk_device, h_tk, sizeof(std::complex<double>) * N * N, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    std::cout << "Error cudaMemcpy" << err << std::endl;
  std::cout << "Done copying h_tk to device" << std::endl;

  cufftHandle plan;
  cufftResult cufft_err = cufftPlan2d(&plan, N, N, CUFFT_Z2Z);
  std::cout << _cudaGetErrorEnum(cufft_err) << std::endl;
  cufft_err = cufftExecZ2Z(plan, h_tk_device, h_k_device, CUFFT_INVERSE); // Inverse FFT: h(k, x, t) = sum(h_tilde(k, x, t) * e^(2pikn / N))
  std::cout << _cudaGetErrorEnum(cufft_err) << std::endl;
  std::cout << "Done executing FFT" << std::endl;

  err = cudaMemcpy(h_k, h_k_device, sizeof(std::complex<double>) * N * N, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
    std::cout << "Error cudaMalloc" << err << std::endl;
  for (int m = 0; m < N; m++) {
    for (int n = 0; n < N; n++) {
      int i = m * N + n;
      std::cout << h_k[i] << std::endl;
    }
  }

  std::cout << "After copy" << std::endl;
  // End wave simulation

  while (!window.should_close ())
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
    glClearColor(0.8, 0.85, 1.0, 1.0);
    glUseProgram(shader_program);

    update(window, clock.tick(), camera);

    glm::mat4 view_projection = camera.get_view_projection();
    glUniformMatrix4fv(view_proj_loc, 1, false, &view_projection[0][0]);

    glUniform1f(time_loc, clock.since_start());

    cube.bind();
    white_texture.bind(0);
    glm::mat4 cube_matrix = glm::translate(glm::identity<glm::mat4>(), glm::vec3(2.0, 0.0, 0.0));
    cube_matrix = glm::rotate(cube_matrix, glm::radians<float>(45.0f), glm::vec3(0.0, 0.0, 1.0));
    glUniformMatrix4fv(model_loc, 1, false, &cube_matrix[0][0]);
    cube.draw();

    water.bind();
    water_normal1.bind(0);
    water_normal2.bind(1);
    for (int x = 0; x < 10; x++) {
      for (int z = 0; z < 10; z++) {
        glm::mat4 water_matrix = glm::translate(glm::identity<glm::mat4>(), glm::vec3(-5.0 + x, 0.0, -7.0 + z));
        //water_matrix = glm::rotate(water_matrix, glm::radians<float>(0.f), glm::vec3(1.f, 0.f, 0.f));
        glUniformMatrix4fv(model_loc, 1, false, &water_matrix[0][0]);
        water.draw();
      }
    }

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

RawModel create_cube() {
   std::vector<float> cube_data = {
    // x, y, z, r, g, b, u, v
    -0.5f, -0.5f, -0.5f, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.5f, -0.5f, -0.5f, 0.0, 1.0, 0.0, 0.0, 0.0,
    -0.5f, 0.5f, -0.5f, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.5f, 0.5f, -0.5f, 0.0, 1.0, 0.0, 0.0, 0.0,
    -0.5f, -0.5f, 0.5f, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.5f, -0.5f, 0.5f, 1.0, 0.0, 0.0, 0.0, 0.0,
    -0.5f, 0.5f, 0.5f, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.5f, 0.5f, 0.5f, 1.0, 0.0, 0.0, 0.0, 0.0,
  };

  std::vector<uint32_t> cube_indices = {
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

  return RawModel(cube_data, cube_indices);
}

static const char *_cudaGetErrorEnum(cufftResult error)
{
  switch (error)
  {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";    
    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";    
    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";    
    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";    
    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";    
    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";    
    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";    
    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";    
    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";    
    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";
}return "<unknown>";
}