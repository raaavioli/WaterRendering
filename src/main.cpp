#include<iostream>
#include<vector>

// External
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#define GL_SILENCE_DEPRECATION
// GL
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Local
#include "ocean.h"
#include "texture.h"
#include "framebuffer.h"
#include "model.h"
#include "camera.h"
#include "clock.h"

struct Window {
public:
  Window(uint32_t width, uint32_t height) {
    if (!glfwInit()) {
      std::cout << "Error: Could not initialize glfw" << std::endl;
      exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL 

    this->window = glfwCreateWindow(width, height, "Water rendering", NULL, NULL);
    if (!this->window) {
      glfwTerminate();
      std::cout << "Could not create glfw window" << std::endl;
      exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(this->window);

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

  inline bool should_close() { return glfwWindowShouldClose(this->window); }
  inline void poll_events() { glfwPollEvents();}
  inline void swap_buffers() { glfwSwapBuffers(this->window);}
  inline GLFWwindow* get_native_window() { return this->window; }

  void resize(Camera& camera) {
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    camera.set_aspect(display_w / (float) display_h);
    glViewport(0, 0, display_w, display_h);
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
  Window window(1260, 1080);
  Clock clock;

  /** ImGui setup begin */
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForOpenGL(window.get_native_window(), true);
  /** TODO: Remove hard coded glsl version */
  ImGui_ImplOpenGL3_Init("#version 330");
  /** ImGui setup end */


  GLuint water_shader_program = create_shader_program(water_vs_code, water_fs_code);
  GLuint skybox_shader_program = create_shader_program(skybox_vs_code, skybox_fs_code);

  /** For rendering a flat water surface
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
  */

  Texture2D white_texture;
  float movement_speed = 1.0;
  float rotation_speed = 30.0;
  Camera camera(glm::vec3(0.0, 1.0, 10.0),
    0.0, 0.0, 45.0f, 1260.0f / 1080.0f, 0.01, 1000.0, 
    rotation_speed, movement_speed
  );

  glEnable(GL_CULL_FACE);
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

  Ocean ocean(200, skybox);

  bool draw_skybox = true;
  bool bind_skybox_cubemap = true;

  int n = 0;
  std::vector<double> times(100);
  while (!window.should_close ()) {
    glClearColor(0.8, 0.85, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

    double dt = clock.tick();
    update(window, dt, camera);
    double start = clock.since_start();
    ocean.update(dt);
    times[n % 100] = (clock.since_start() - start) * 1000;
    n++;

    ocean.draw(water_shader_program, camera);

    /** SKYBOX RENDERING BEGIN (done last) **/
    if (draw_skybox) {
      glm::mat4 skybox_view_projection = camera.get_view_projection(false);
      glUseProgram(skybox_shader_program);
      glUniformMatrix4fv(skybox_view_proj_loc, 1, false, &skybox_view_projection[0][0]);
      skybox.draw();
    }
    /** SKYBOX RENDERING END **/

    double sum = 0.0;
    for (double t : times)
      sum += t;

    /** GUI RENDERING BEGIN **/
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::Begin("Settings panel");
    ImGui::Text("FPS: %f", (1 / dt));
    ImGui::Text("Update-time: %f (ms)", (sum / 100.0));
    ImGui::Text("Simulation");
    ImGui::Dummy(ImVec2(0.0, 5.0));
    //ImGui::SliderFloat("Simulation speed", &simulation_speed, 0.0, 10.0);
    //ImGui::SliderInt("Num tiles", &num_tiles, 1, 20);
    ImGui::Dummy(ImVec2(0.0, 15.0));

    ImGui::Text("Wave");
    ImGui::Dummy(ImVec2(0.0, 5.0));

    ImGui::Dummy(ImVec2(0.0, 15.0));

    ImGui::Text("Environment");
    ImGui::Dummy(ImVec2(0.0, 5.0));
    ImGui::Checkbox("Skybox", &draw_skybox);
    ImGui::Checkbox("Cubemap", &bind_skybox_cubemap);
    ImGui::Dummy(ImVec2(0.0, 15.0));

    ImGui::Text("Camera");
    ImGui::Dummy(ImVec2(0.0, 5.0));
    ImGui::SliderFloat("Movement speed", &movement_speed, 1.0, 10.0);
    ImGui::SliderFloat("Rotation speed", &rotation_speed, 1.0, 200.0);
    ImGui::Dummy(ImVec2(0.0, 15.0));
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    /** GUI RENDERING END **/

    camera.set_movement_speed(movement_speed);
    camera.set_rotation_speed(rotation_speed);

    window.resize(camera);
    window.swap_buffers ();
    window.poll_events ();
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  return 0;
}

void update(const Window& window, double dt, Camera& camera) {
  // std::cout << "Frame rate: " << 1.0 / dt << " FPS" << std::endl;

  if (window.is_key_pressed(GLFW_KEY_LEFT)) camera.rotate_yaw(dt);
  if (window.is_key_pressed(GLFW_KEY_RIGHT)) camera.rotate_yaw(-dt);
  if (window.is_key_pressed(GLFW_KEY_UP)) camera.rotate_pitch(dt);
  if (window.is_key_pressed(GLFW_KEY_DOWN)) camera.rotate_pitch(-dt);

  int direction = 0;
  if (window.is_key_pressed(GLFW_KEY_W)) direction |= Camera::FORWARD;
  if (window.is_key_pressed(GLFW_KEY_S)) direction |= Camera::BACKWARD;
  if (window.is_key_pressed(GLFW_KEY_D)) direction |= Camera::RIGHT;
  if (window.is_key_pressed(GLFW_KEY_A)) direction |= Camera::LEFT;
  camera.move(dt, direction);
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

