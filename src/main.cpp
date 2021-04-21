#include<iostream>
#include<vector>

#define GL_SILENCE_DEPRECATION

// CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cufft.h>

#define CUDA_ASSERT(err) if(err != cudaSuccess) std::cout << "Cuda Error: " << err << ", Line: " << __LINE__ << std::endl;
#define CUFFT_ASSERT(err) if(err != CUFFT_SUCCESS) std::cout << "Cufft Error: " << err << ", Line: " << __LINE__ << std::endl;

// GL
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Local
#include "wave_utils.h"
#include "texture.h"
#include "framebuffer.h"

struct Vertex {
  glm::vec3 position;
  glm::vec3 color;
  glm::vec3 normal;
  glm::vec2 uv;
};

/** STRUCTS */
struct RawModel {
  RawModel(const std::vector<Vertex>& data, const std::vector<uint32_t>& indices, GLenum usage) : gl_usage(usage) {
    assert((indices.size() % 3) == 0);
    int vertex_size = sizeof(Vertex);

    glGenVertexArrays(1, &this->renderer_id);
    glBindVertexArray(this->renderer_id);
    glGenBuffers(1, &this->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
    glBufferData(GL_ARRAY_BUFFER, data.size() * vertex_size, &data[0], usage);
    glGenBuffers(1, &this->ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint32_t), &indices[0], GL_STATIC_DRAW);
    this->index_count = indices.size();

    // Bind buffers to VAO
    bind();
    glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
    glEnableVertexAttribArray(0); // Position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, (const void*) offsetof(Vertex, position));
    glEnableVertexAttribArray(1); // Color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_size, (const void*) offsetof(Vertex, color));
    glEnableVertexAttribArray(2); // Normal
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, vertex_size, (const void*) offsetof(Vertex, normal));
    glEnableVertexAttribArray(3); // UV
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, vertex_size, (const void*) offsetof(Vertex, uv));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->ebo);
    unbind();
  };

  ~RawModel() {
    // Wish to delete buffers here but may cause problems
    /*glDeleteBuffers(1, ebo);
    glDeleteBuffers(1, vbo);
    glDeleteBuffers(1, renderer_id);*/
  }

  void update_vertex_data(const std::vector<Vertex>& vertices) {
    if (this->gl_usage == GL_DYNAMIC_DRAW || this->gl_usage == GL_STREAM_DRAW) {
      this->bind();
      glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(Vertex), &vertices[0]);
      this->unbind();
    } else {
      std::cout << "ERROR (update_data): Usage of model data has to be GL_DYNAMIC_DRAW or GL_STREAM_DRAW" << std::endl;
    }
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

  GLenum gl_usage;

  uint32_t index_count;
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
GLuint create_shader_program(const char* vs_code, const char* fs_code);
GLuint create_model_vao(float* data, size_t data_size, uint32_t* indices, size_t indices_size);
void update_surface_vertices(uint32_t Nplus1, std::vector<Vertex>& vertices, const std::vector<glm::vec3>& origin_positions,
  std::complex<double>* displacement_x, std::complex<double>* displacement_y, std::complex<double>* displacement_z,
  std::complex<double>* gradient_x, std::complex<double>* gradient_y);

const char* texture_vs_code = R"(
#version 410 core
layout(location = 0) in vec2 a_Pos;
layout(location = 1) in vec2 a_TextureData;

layout(location = 0) out vec2 vs_UV;

void main() {
  gl_Position = vec4(a_Pos, 0.0, 1.0); 
}
)";

const char* texture_fs_code = R"(
#version 410 core
out vec4 color;

layout(location = 0) in vec2 vs_UV;

uniform sampler2D texture0;

void main() {
  color = texture(texture0, vs_UV);
}
)";

const char* vertex_shader_code = R"(
#version 410 core
layout(location = 0) in vec3 a_Pos;
layout(location = 1) in vec3 a_Color;
layout(location = 2) in vec3 a_Normal;
layout(location = 3) in vec2 a_UV;

layout(location = 0) out vec3 vs_Color;
layout(location = 1) out vec2 vs_UV;
layout(location = 2) out float vs_Time;
layout(location = 3) out vec3 vs_Normal;
layout(location = 4) out vec3 vs_LightDir;
layout(location = 5) out vec3 vs_CameraDir;
layout(location = 6) out vec3 vs_Pos;

uniform mat4 u_ViewProjection;
uniform mat4 u_Model;
uniform float u_Time;

void main()
{
  vs_Time = u_Time;
  vs_Color = a_Color;
  vs_UV = a_UV;
  vs_Normal = (u_Model * vec4(a_Normal, 0.0)).xyz;
  vs_LightDir = vec3(0.0, 1.0, 0.0);
  vs_Pos = a_Pos;

  vec4 m_Pos = u_Model * vec4(a_Pos, 1.0);
  vs_CameraDir = normalize(-m_Pos.xyz);
  gl_Position = u_ViewProjection * m_Pos; 
}
)";

const char* fragment_shader_code = R"(
#version 410 core
out vec4 color;

layout(location = 0) in vec3 vs_Color;
layout(location = 1) in vec2 vs_UV;
layout(location = 2) in float vs_Time;
layout(location = 3) in vec3 vs_Normal;
layout(location = 4) in vec3 vs_LightDir;
layout(location = 5) in vec3 vs_CameraDir;
layout(location = 6) in vec3 vs_Pos;

uniform sampler2D texture0;
//uniform sampler2D texture1;

void main()
{
  // Blinn-Phong illumination using half-way vector instead of reflection.
  vec3 light_color = vec3(1.0, 1.0, 1.0);
  vec3 halfwayDir = normalize(vs_LightDir + vs_CameraDir);  
  float specular = pow(max(dot(vs_Normal, halfwayDir), 0.0), 20.0);
  float diffuse = max(dot(vs_Normal, vs_LightDir), 0);
  
  // Trip mode on.
  //float height = normalize(vs_Pos).y;
  //float xx = normalize(vs_Pos).x + (int(vs_Time) % 1000) / 1000.0f * normalize(vs_Pos).z;
  //vec3 rand_color = vec3(0.8, 0.9, 1.0) - vec3(2.3*sin(height + vs_Time + xx), 13*cos(height + vs_Time - xx), 5*sin(height + vs_Time + xx)); 
  //color = vec4(diffuse * rand_color + specular * light_color, 1.0);
  
  color = vec4(diffuse * vs_Color + specular * light_color, 1.0);
  //color = texture(texture0, vs_UV); 
}
)";

int main(void)
{
  Window window;
  Clock clock;

  GLuint shader_program = create_shader_program(vertex_shader_code, fragment_shader_code);
  glUseProgram(shader_program);

  std::vector<Vertex> square_data = {
    Vertex{glm::vec3(-0.5, 0.0, -0.5), glm::vec3(0.6, 0.6, 0.9), glm::vec3(0.0, 1.0, 0.0), glm::vec2(1.0, 1.0)}, 
    Vertex{glm::vec3(0.5, 0, -0.5), glm::vec3(0.6, 0.6, 0.9), glm::vec3(0.0, 1.0, 0.0), glm::vec2(0.0, 1.0)},
    Vertex{glm::vec3(-0.5, 0, 0.5), glm::vec3(0.6, 0.6, 0.9), glm::vec3(0.0, 1.0, 0.0), glm::vec2(1.0, 0.0)},  
    Vertex{glm::vec3(0.5,  0, 0.5), glm::vec3(0.6, 0.6, 0.9), glm::vec3(0.0, 1.0, 0.0), glm::vec2(0.0, 0.0)},
  };

  std::vector<uint32_t> square_indices = {
    1, 0, 2, 
    2, 3, 1
  };

  RawModel water(square_data, square_indices, GL_STATIC_DRAW);
  Texture white_texture;
  Camera camera = { .position = glm::vec3(0.0, 3.0, 1.0),
	  .yaw = 0.0, .pitch = 0.0,
    .fovy = 45.0f, 1260.0f / 1080.0f, 0.01, 1000.0
  };

  glEnable(GL_CULL_FACE);

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS); 

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Wire frame
  //glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );

  GLuint view_proj_loc = glGetUniformLocation(shader_program, "u_ViewProjection");
  GLuint model_loc = glGetUniformLocation(shader_program, "u_Model");
  GLuint time_loc = glGetUniformLocation(shader_program, "u_Time");
  GLuint tex0_loc = glGetUniformLocation(shader_program, "texture0");
  GLuint tex1_loc  = glGetUniformLocation(shader_program, "texture1");

  glUniform1i(tex0_loc, 0);
  glUniform1i(tex1_loc, 1);

  // Wave simulation
  const int N = 256;
  const int Nplus1 = N + 1;
  double length = 300;
  double two_pi = glm::two_pi<double>();

  std::vector<Vertex> vertices(Nplus1 * Nplus1);
  std::vector<glm::vec3> origin_positions(Nplus1 * Nplus1);
  std::vector<uint32_t> indices;
  indices.reserve(N * N * 6);
  float tile_dim = 5.0;
  for (int z = 0; z < Nplus1; z++) {
    for (int x = 0; x < Nplus1; x++) {
      int i0 = z * Nplus1 + x;
      Vertex vertex;
      vertex.position = glm::vec3(-0.5 + x * tile_dim / float(N), 0, -0.5 + z * tile_dim / float(N));
      vertex.color = glm::vec3(0.1, 0.3, 0.5);
      vertices[i0] = vertex;
      origin_positions[i0] = vertex.position;

      if (x < N && z < N) {
        int i1 = (z + 1) * Nplus1 + x;
        int i2 = (z + 1) * Nplus1 + (x + 1);
        int i3 = z * Nplus1 + (x + 1);
        indices.push_back(i3);
        indices.push_back(i0);
        indices.push_back(i1);
        indices.push_back(i1);
        indices.push_back(i2);
        indices.push_back(i3);
      }
    }
  }

  RawModel water_surface(vertices, indices, GL_DYNAMIC_DRAW);

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

  std::complex<double>* h_tk = new std::complex<double>[N * N]; // h_tilde(k, x, t)
  std::complex<double>* h_k = new std::complex<double>[N * N]; // h(k, x, t)
  std::complex<double>* h_k_disp_x = new std::complex<double>[N * N]; // x-displacement of h(k, x, t)
  std::complex<double>* h_k_disp_z = new std::complex<double>[N * N]; // z-displacement of h(k, x, t)
  std::complex<double>* dh_k_dx = new std::complex<double>[N * N]; // x-gradient of h(k, x, t)
  std::complex<double>* dh_k_dz = new std::complex<double>[N * N]; // z-gradient of h(k, x, t)

  cufftDoubleComplex* h_tk_device;
  cufftDoubleComplex* h_k_device;
  cufftDoubleComplex* h_k_disp_x_device;
  cufftDoubleComplex* h_k_disp_z_device;
  cufftDoubleComplex* dh_k_dx_device;
  cufftDoubleComplex* dh_k_dz_device;

  CUDA_ASSERT(cudaMalloc ((void**) &h_tk_device, sizeof(std::complex<double>) * N * N));
  CUDA_ASSERT(cudaMalloc ((void**) &h_k_device, sizeof(std::complex<double>) * N * N));
  CUDA_ASSERT(cudaMalloc ((void**) &h_k_disp_x_device, sizeof(std::complex<double>) * N * N));
  CUDA_ASSERT(cudaMalloc ((void**) &h_k_disp_z_device, sizeof(std::complex<double>) * N * N));
  CUDA_ASSERT(cudaMalloc ((void**) &dh_k_dx_device, sizeof(std::complex<double>) * N * N));
  CUDA_ASSERT(cudaMalloc ((void**) &dh_k_dz_device, sizeof(std::complex<double>) * N * N));

  cufftHandle plan;
  CUFFT_ASSERT(cufftPlan2d(&plan, N, N, CUFFT_Z2Z));

  Image displacement_image(N, N, GL_RGB);

  uint32_t displacement_id = 0;

  while (!window.should_close ()) {
    // Setup h_tk + device
    for (int m = 0; m < N; m++) {
      for (int n = 0; n < N; n++) {
        int i = m * N + n;
        float kx = (n - N / 2.f) * two_pi / length;
        float kz = (m - N / 2.f) * two_pi / length;
        glm::vec2 K(kx, kz);
        h_tk[i] = h_tilde(h0_tk[i], h0_tmk[i], K, clock.since_start());
        dh_k_dx[i] = h_tk[i] * std::complex<double>(0.0, kx);
        dh_k_dz[i] = h_tk[i] * std::complex<double>(0.0, kz);
        double k_length = glm::length(K);
        if (k_length > 0.00001) {
          h_k_disp_x[i] = h_tk[i] * std::complex<double>(0.0, -kx / k_length);
          h_k_disp_z[i] = h_tk[i] * std::complex<double>(0.0, -kz / k_length);
        } else {
          h_k_disp_x[i] = h_tk[i] * std::complex<double>(0.0, 0.0);
          h_k_disp_z[i] = h_tk[i] * std::complex<double>(0.0, 0.0);
        }
      }
    }

    CUDA_ASSERT(cudaMemcpy(h_tk_device, h_tk, sizeof(std::complex<double>) * N * N, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(h_k_disp_x_device, h_k_disp_x, sizeof(std::complex<double>) * N * N, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(h_k_disp_z_device, h_k_disp_z, sizeof(std::complex<double>) * N * N, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(dh_k_dx_device, dh_k_dx, sizeof(std::complex<double>) * N * N, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(dh_k_dz_device, dh_k_dz, sizeof(std::complex<double>) * N * N, cudaMemcpyHostToDevice));
    // Inverse FFT: h(k, x, t) = sum(h_tilde(k, x, t) * e^(2pikn / N))
    CUFFT_ASSERT(cufftExecZ2Z(plan, h_tk_device, h_k_device, CUFFT_INVERSE)); 
    // In place transforms
    CUFFT_ASSERT(cufftExecZ2Z(plan, h_k_disp_x_device, h_k_disp_x_device, CUFFT_INVERSE)); 
    CUFFT_ASSERT(cufftExecZ2Z(plan, h_k_disp_z_device, h_k_disp_z_device, CUFFT_INVERSE)); 
    CUFFT_ASSERT(cufftExecZ2Z(plan, dh_k_dx_device, dh_k_dx_device, CUFFT_INVERSE)); 
    CUFFT_ASSERT(cufftExecZ2Z(plan, dh_k_dz_device, dh_k_dz_device, CUFFT_INVERSE)); 

    CUDA_ASSERT(cudaMemcpy(h_k, h_k_device, sizeof(std::complex<double>) * N * N, cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(h_k_disp_x, h_k_disp_x_device, sizeof(std::complex<double>) * N * N, cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(h_k_disp_z, h_k_disp_z_device, sizeof(std::complex<double>) * N * N, cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(dh_k_dx, dh_k_dx_device, sizeof(std::complex<double>) * N * N, cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(dh_k_dz, dh_k_dz_device, sizeof(std::complex<double>) * N * N, cudaMemcpyDeviceToHost));

    for (int m = 0; m < N; m++) {
      for (int n = 0; n < N; n++) {
        int i = m * N + n;
        int sign = (m + n) % 2 == 0 ? 1 : -1; // Larsson (2012), Equation 4.6
        h_k[i] /= sign * (N * N);
        h_k_disp_x[i] /= sign * (N * N);
        h_k_disp_z[i] /= sign * (N * N);
        dh_k_dx[i] /= sign * (N * N);
        dh_k_dz[i] /= sign * (N * N);
      }
    }

    if (!displacement_id) {
      double min = 5;
      double max = -5;
      for (int m = 0; m < N; m++) {
        for (int n = 0; n < N; n++) {
          int i = m * N + n;
          if (h_k[i].real() < min)
            min = h_k[i].real();
          if (h_k[i].real() > max)
            max = h_k[i].real();
        }
      }

      for (int m = 0; m < N; m++) {
        for (int n = 0; n < N; n++) {
          int i = m * N + n;
          float value = (h_k[i].real() - min) / (max - min);
          char color = 255 * value;
          displacement_image.set_pixel(n, m, color, color, color);
        }
      }
      Texture displacement_tex(displacement_image);
      displacement_id = displacement_tex.get_texture_id();
    }

    glClearColor(0.8, 0.85, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
    glUseProgram(shader_program);

    update(window, clock.tick(), camera);

    glm::mat4 view_projection = camera.get_view_projection();
    glUniformMatrix4fv(view_proj_loc, 1, false, &view_projection[0][0]);

    glUniform1f(time_loc, clock.since_start());


    //for (int y = 0; y < 10; y++) {
      //for (int x = 0; x < 10; x++) {
        /*glm::mat4 water_matrix = glm::translate(glm::identity<glm::mat4>(), glm::vec3(0.0, 0.0, 0.0));
        water_matrix = glm::rotate(water_matrix, glm::radians<float>(90), glm::vec3(1.0, 0.0, 0.0));
        glUniformMatrix4fv(model_loc, 1, false, &water_matrix[0][0]);
        water.bind();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, displacement_id);
        water.draw();*/
      //}
    //}


    glm::mat4 water_matrix = glm::identity<glm::mat4>();
    water_matrix = glm::rotate(water_matrix, glm::radians<float>(0), glm::vec3(1.0, 0.0, 0.0));
    glUniformMatrix4fv(model_loc, 1, false, &water_matrix[0][0]);

    update_surface_vertices(Nplus1, vertices, origin_positions, h_k, h_k_disp_x, h_k_disp_z, dh_k_dz, dh_k_dz);
    water_surface.update_vertex_data(vertices);

    water_surface.bind();
    int num_tiles = 10;
    for (int z = 0; z < num_tiles; z++) {
      for (int x = 0; x < num_tiles; x++) {
        glm::mat4 water_matrix = glm::translate(glm::identity<glm::mat4>(), 
          glm::vec3(tile_dim * (-num_tiles / 2.0f + x), 0.0, tile_dim * (-num_tiles / 2.0f + z))
        );
        //water_matrix = glm::rotate(water_matrix, glm::radians<float>(0), glm::vec3(1.0, 0.0, 0.0));
        glUniformMatrix4fv(model_loc, 1, false, &water_matrix[0][0]);
        water_surface.draw();
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

GLuint create_shader_program(const char* vs_code, const char* fs_code) {
  int success;
  char infoLog[512];

  GLuint vs = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vs, 1, &vs_code, NULL);
  glCompileShader(vs);
  glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
  if(!success) {
    glGetShaderInfoLog(vs, sizeof(infoLog), NULL, infoLog);
    std::cout << "ERROR: Vertex shader compilation failed\n" << infoLog << std::endl;
  };

  GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fs, 1, &fs_code, NULL);
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

void update_surface_vertices(uint32_t Nplus1, std::vector<Vertex>& vertices, const std::vector<glm::vec3>& origin_positions,
  std::complex<double>* displacement_y, std::complex<double>* displacement_x, std::complex<double>* displacement_z,
  std::complex<double>* gradient_x, std::complex<double>* gradient_y) {
  uint32_t N = Nplus1 - 1;
  for (int z = 0; z < Nplus1; z++) {
    for (int x = 0; x < Nplus1; x++) {
      int i_v = z * Nplus1 + x;
      int i_d = (z % N) * N + x % N;
      glm::vec3 displacement(displacement_x[i_d].real(), displacement_y[i_d].real(), displacement_z[i_d].real());
      vertices[i_v].position = origin_positions[i_v] + displacement; 
    }
  }


  // Add normals for all 8 triangles connecting to every vertex. Then normalize the result.
  for (int z = 0; z < Nplus1; z++) {
    for (int x = 0; x < Nplus1; x++) {
      glm::vec3 sum_normals(0.0, 0.0, 0.0);
      int i = z * Nplus1 + x;
      glm::vec3 middle = vertices[i].position;
      if (x > 0) {
        glm::vec3 left = vertices[i - 1].position;
        if (z > 0) {
          glm::vec3 bottom = vertices[i - Nplus1].position;
          glm::vec3 bottom_left = vertices[i - 1 - Nplus1].position;
          glm::vec3 v1 = glm::normalize(bottom - middle);
          glm::vec3 v2 = glm::normalize(bottom_left - middle);
          sum_normals += glm::cross(v1, v2);
          
          v1 = glm::normalize(bottom_left - middle);
          v2 = glm::normalize(left - middle);
          sum_normals += glm::cross(v1, v2);
        }
        if (z < Nplus1 - 1) {
          glm::vec3 top_left = vertices[i - 1 + Nplus1].position;
          glm::vec3 v1 = glm::normalize(left - middle);
          glm::vec3 v2 = glm::normalize(top_left - middle);
          sum_normals += glm::cross(v1, v2);

          glm::vec3 top = vertices[i + Nplus1].position;
          v1 = glm::normalize(top_left - middle);
          v2 = glm::normalize(top - middle);
          sum_normals += glm::cross(v1, v2);
        }
      }
      if (x < Nplus1 - 1) {
        glm::vec3 right = vertices[i + 1].position;
        if (z < Nplus1 - 1) {
          glm::vec3 top = vertices[i + Nplus1].position;
          glm::vec3 top_right = vertices[i + 1 + Nplus1].position;
          glm::vec3 v1 = glm::normalize(top - middle);
          glm::vec3 v2 = glm::normalize(top_right - middle);
          sum_normals += glm::cross(v1, v2);

          v1 = glm::normalize(top_right - middle);
          v2 = glm::normalize(right - middle);
          sum_normals += glm::cross(v1, v2);
        }
        if (z > 0) {
          glm::vec3 bottom_right = vertices[i + 1 - Nplus1].position;
          glm::vec3 v1 = glm::normalize(right - middle);
          glm::vec3 v2 = glm::normalize(bottom_right - middle);
          sum_normals += glm::cross(v1, v2);

          glm::vec3 bottom = vertices[i - Nplus1].position;
          v1 = glm::normalize(bottom_right - middle);
          v2 = glm::normalize(bottom - middle);
          sum_normals += glm::cross(v1, v2);
        }
      }

      vertices[i].normal = glm::normalize(sum_normals);
    }
  }
}