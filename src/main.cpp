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
#include "model.h"




struct Camera {
	glm::vec3 position;
	float yaw, pitch;
  float fovy, aspect, near, far;

	glm::mat4 get_view_projection(bool translate) const {
    
    glm::mat4 rotation = get_rotation();

    glm::mat4 translation = glm::mat4(1.0);
    if (translate)
      translation = glm::translate(translation, position);

    glm::mat4 projection = glm::perspective(this->fovy, this->aspect, this->near, this->far);
		return projection * glm::inverse(translation * rotation);
	}

  glm::mat4 get_rotation() const {
     glm::mat4 rotate = glm::rotate(glm::mat4(1.0), 
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

const char* skybox_vs_code = R"(
#version 410 core
layout(location = 0) in vec3 a_Pos;

layout(location = 0) out vec3 vs_TexCoord;

uniform mat4 u_ViewProjection;

void main() {
  vs_TexCoord = a_Pos;
  vec4 proj_pos = u_ViewProjection * vec4(a_Pos, 1.0);
  gl_Position = proj_pos.xyww;  
}
)";

const char* skybox_fs_code = R"(
#version 410 core
out vec4 color;

layout(location = 0) in vec3 vs_TexCoord;

uniform samplerCube cube_map;

void main() {
  color = texture(cube_map, vs_TexCoord);
}
)";

const char* water_vs_code = R"(
#version 410 core
layout(location = 0) in vec3 a_Pos;
layout(location = 1) in vec3 a_Color;
layout(location = 2) in vec3 a_Normal;
layout(location = 3) in vec2 a_UV;

layout(location = 0) out vec3 vs_Color;
layout(location = 1) out float vs_Time;
layout(location = 2) out vec3 vs_Normal;
layout(location = 3) out vec3 vs_LightSourceDir;
layout(location = 4) out vec3 vs_CameraDir;

uniform mat4 u_ViewProjection;
uniform mat4 u_Model;
uniform float u_Time;
uniform vec3 u_CameraPos;

void main()
{
  // Constants
  vec3 lightPos = vec3(20.0, 30.0, -30.0);
 
  vs_Color = a_Color;
  vs_Time = u_Time;

  vs_Normal = normalize(transpose(inverse(mat3(u_Model))) * a_Normal);

  vec4 m_Pos = u_Model * vec4(a_Pos, 1.0);
  vs_LightSourceDir = normalize(lightPos - m_Pos.xyz);
  vs_CameraDir = normalize(u_CameraPos - m_Pos.xyz);
  gl_Position = u_ViewProjection * m_Pos; 
}
)";

const char* water_fs_code = R"(
#version 410 core
out vec4 color;

layout(location = 0) in vec3 vs_Color;
layout(location = 1) in float vs_Time;
layout(location = 2) in vec3 vs_Normal;
layout(location = 3) in vec3 vs_LightSourceDir;
layout(location = 4) in vec3 vs_CameraDir;

uniform sampler2D texture0;
uniform samplerCube cube_map;

// https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel
// Tessendorf 4.3 - Building a Shader for renderman
// @param I Incident vector
// @param N Normal vector
// @param ior Index of refraction
float reflectivity(vec3 I, vec3 N, float ior) {
  float costhetai = abs(dot(normalize(I), normalize(N)));
  float thetai = acos(costhetai);
  float sinthetat = sin(thetai) / ior;
  float thetat = asin(sinthetat);
  if(thetai == 0.0) {
    float reflectivity = (ior - 1)/(ior + 1);
    return reflectivity * reflectivity;
  } else {
    float fs = sin(thetat - thetai) / sin(thetat + thetai);
    float ts = tan(thetat - thetai) / tan(thetat + thetai);
    return 0.5 * ( fs*fs + ts*ts );
  } 
}

void main() { 
  // Blinn-Phong illumination using half-way vector instead of reflection.
  float refraction_index = 1.0 / 1.33;
  vec3 refractionDir = refract(-vs_CameraDir, vs_Normal, refraction_index);
  vec3 reflectionDir = reflect(-vs_CameraDir, vs_Normal);
  float reflectivity = reflectivity(vs_CameraDir, vs_Normal, 1.0 / refraction_index);

  // Intensities
  vec3 i_reflect = texture(cube_map, reflectionDir).xyz; 
  vec3 i_refract = vs_Color;
  if (refractionDir != vec3(0.0)) // If refractionDir is 0-vector, something is wrong. 
    i_refract = texture(cube_map, refractionDir).xyz;

  vec3 halfwayDir = normalize(vs_LightSourceDir + vs_CameraDir);
  float specular = pow(max(dot(vs_Normal, halfwayDir), 0.0), 10.0);
  
  const vec3 light_color = 0.4 * vec3(1.0);
  
  vec3 reflection_refraction = reflectivity * i_reflect + (1 - reflectivity) * i_refract;
  color = vec4(reflection_refraction + light_color * specular, 1.0);
}
)";

int main(void)
{
  Window window;
  Clock clock;

  GLuint water_shader_program = create_shader_program(water_vs_code, water_fs_code);
  GLuint skybox_shader_program = create_shader_program(skybox_vs_code, skybox_fs_code);

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
  Texture2D white_texture;
  Camera camera = { .position = glm::vec3(0.0, 0.1, 10.0),
	  .yaw = 15.0, .pitch = 0.0,
    .fovy = 45.0f, 1260.0f / 1080.0f, 0.01, 1000.0
  };

  //glEnable(GL_CULL_FACE);

  glEnable(GL_DEPTH_TEST);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Wire frame
  //glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );

  // Skybox Shader Uniforms
  GLuint skybox_view_proj_loc = glGetUniformLocation(skybox_shader_program, "u_ViewProjection");
  GLuint skybox_texture_loc = glGetUniformLocation(skybox_shader_program, "cube_map");
  glUseProgram(skybox_shader_program);
  glUniform1f(skybox_texture_loc, 0);

  // Textures received from: https://www.humus.name/index.php?page=Textures
  const char* skansen_folder = "skansen";
  const char* ocean_folder = "ocean";
  const char* church_folder = "church";
  Skybox skybox(skansen_folder, true);
  //Skybox desert_skybox(desert_cubemap_filename);
  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

  // Water Shader Uniforms
  GLuint shader_view_proj_loc = glGetUniformLocation(water_shader_program, "u_ViewProjection");
  GLuint model_loc = glGetUniformLocation(water_shader_program, "u_Model");
  GLuint time_loc = glGetUniformLocation(water_shader_program, "u_Time");
  GLuint camera_pos_loc = glGetUniformLocation(water_shader_program, "u_CameraPos");
  GLuint tex0_loc = glGetUniformLocation(water_shader_program, "texture0");
  GLuint shader_cube_map_loc  = glGetUniformLocation(water_shader_program, "cube_map");

  glUseProgram(water_shader_program);
  glUniform1i(tex0_loc, 0);
  glUniform1i(shader_cube_map_loc, 1);

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
      vertex.color = glm::vec3(1.0, 0.0, 0.0); //glm::vec3(52 / 255.0, 155 / 255.0, 235 / 255.0);
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

  float simulation_speed = 2.0;

  while (!window.should_close ()) {
    // Setup h_tk + device
    for (int m = 0; m < N; m++) {
      for (int n = 0; n < N; n++) {
        int i = m * N + n;
        float kx = (n - N / 2.f) * two_pi / length;
        float kz = (m - N / 2.f) * two_pi / length;
        glm::vec2 K(kx, kz);
        h_tk[i] = h_tilde(h0_tk[i], h0_tmk[i], K, simulation_speed * clock.since_start());
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
      Texture2D displacement_tex(displacement_image);
      displacement_id = displacement_tex.get_texture_id();
    }

    glClearColor(0.8, 0.85, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

    double time = clock.tick();
    update(window, time, camera);
    //std::cout << 1 / time << " FPS" << std::endl;

    /** WAVE RENDERING BEGIN **/
    glUseProgram(water_shader_program);
    glm::mat4 wave_view_projection = camera.get_view_projection(true);
    glUniformMatrix4fv(shader_view_proj_loc, 1, false, &wave_view_projection[0][0]);
    glUniform1f(time_loc, clock.since_start());
    glUniform3f(camera_pos_loc, camera.position.x, camera.position.y, camera.position.z);

    //for (int y = 0; y < 10; y++) {
      //for (int x = 0; x < 10; x++) {
        /*glm::mat4 water_matrix = glm::translate(glm::identity<glm::mat4>(), glm::vec3(0.0, 0.0, 0.0));
        water_matrix = glm::rotate(water_matrix, glm::radians<float>(0), glm::vec3(1.0, 0.0, 0.0));
        water_matrix = glm::scale(water_matrix, glm::vec3(10.0, 1.0, 10.0));
        glUniformMatrix4fv(model_loc, 1, false, &water_matrix[0][0]);
        water.bind();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, displacement_id);
        skybox.bind_cube_map(1);
        water.draw();
        skybox.unbind_cube_map();*/
      //}
    //}

    glm::mat4 water_matrix = glm::identity<glm::mat4>();
    water_matrix = glm::rotate(water_matrix, glm::radians<float>(90), glm::vec3(1.0, 0.0, 0.0));
    glUniformMatrix4fv(model_loc, 1, false, &water_matrix[0][0]);

    update_surface_vertices(Nplus1, vertices, origin_positions, h_k_disp_x, h_k, h_k_disp_z, dh_k_dx, dh_k_dz);
    water_surface.update_vertex_data(vertices);

    water_surface.bind();
    skybox.bind_cube_map(1);
    //desert_skybox.bind_cube_map(1);
    int num_tiles = 10;
    for (int z = 0; z < num_tiles; z++) {
      for (int x = 0; x < num_tiles; x++) {
        glm::mat4 water_matrix = glm::translate(glm::identity<glm::mat4>(), 
          glm::vec3(tile_dim * (-num_tiles / 2.0f + x), 0.0, -3.0 + tile_dim * (-num_tiles / 2.0f + z))
        );
        //water_matrix = glm::rotate(water_matrix, glm::radians<float>(90), glm::vec3(1.0, 0.0, 0.0));
        glUniformMatrix4fv(model_loc, 1, false, &water_matrix[0][0]);
        water_surface.draw();
      }
    }

    /** WAVE RENDERING END **/

    /** SKYBOX RENDERING BEGIN (done last) **/
    glm::mat4 skybox_view_projection = camera.get_view_projection(false);
    glUseProgram(skybox_shader_program);
    glUniformMatrix4fv(skybox_view_proj_loc, 1, false, &skybox_view_projection[0][0]);
    skybox.draw();
    //desert_skybox.draw();

    /** SKYBOX RENDERING END **/

    window.swap_buffers ();
    window.poll_events ();
  }
  return 0;
}

void update(const Window& window, double dt, Camera& camera) {
  // std::cout << "Frame rate: " << 1.0 / dt << " FPS" << std::endl;

  float rotation_speed = 10.f;

  if (window.is_key_pressed(GLFW_KEY_LEFT)) {
    camera.yaw += rotation_speed * dt;
  }
  if (window.is_key_pressed(GLFW_KEY_RIGHT)) {
    camera.yaw -= rotation_speed * dt;
  }
  if (window.is_key_pressed(GLFW_KEY_UP)) {
    camera.pitch += rotation_speed * dt;
  }
  if (window.is_key_pressed(GLFW_KEY_DOWN)) {
    camera.pitch -= rotation_speed * dt;
  }

  glm::vec3 forward = glm::vec3(camera.get_rotation() * glm::vec4(0.0, 0.0, -1.0, 0.0));
  glm::vec3 right = glm::vec3(camera.get_rotation() * glm::vec4(1.0, 0.0, 0.0, 0.0));

  float movement_speed = 0.5;
  if (window.is_key_pressed(GLFW_KEY_W)) {
    camera.position += forward * (float) (dt * movement_speed);
  }
  if (window.is_key_pressed(GLFW_KEY_S)) {
    camera.position -= forward * (float) (dt * movement_speed);
  }
  if (window.is_key_pressed(GLFW_KEY_D)) {
    camera.position += right * (float) (dt * movement_speed);
  }
  if (window.is_key_pressed(GLFW_KEY_A)) {
    camera.position -= right * (float) (dt * movement_speed);
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
  std::complex<double>* displacement_x, std::complex<double>* displacement_y, std::complex<double>* displacement_z,
  std::complex<double>* gradient_x, std::complex<double>* gradient_z) {
  uint32_t N = Nplus1 - 1;
  for (uint32_t z = 0; z < Nplus1; z++) {
    for (uint32_t x = 0; x < Nplus1; x++) {
      int i_v = z * Nplus1 + x;
      int i_d = (z % N) * N + x % N;
      double lambda = -1.0;
      glm::vec3 displacement(lambda * displacement_x[i_d].real(), displacement_y[i_d].real(), lambda * displacement_z[i_d].real());
      vertices[i_v].position = origin_positions[i_v] + displacement;
    }
  }

  for (uint32_t z = 0; z < Nplus1; z++) {
    for (uint32_t x = 0; x < Nplus1; x++) {
      int i_v = z * Nplus1 + x;
      int i_d = (z % N) * N + x % N;
      double ex = gradient_x[i_d].real();
      double ez = gradient_z[i_d].real();
      vertices[i_v].normal = glm::vec3(
        -ex * 5.0, 
        1.0, 
        -ez * 5.0);
    }
  }
}