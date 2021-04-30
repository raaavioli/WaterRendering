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
#include "shader.h"

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

int main(void)
{
  Window window(2000, 1300);
  Clock clock;

  /** ImGui setup begin */
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForOpenGL(window.get_native_window(), true);
  /** TODO: Remove hard coded glsl version */
  ImGui_ImplOpenGL3_Init("#version 330");
  /** ImGui setup end */

  GLuint color_water_shader = create_shader_program(water_vs_code, color_water_fs_code);
  GLuint cubemap_water_shader = create_shader_program(water_vs_code, cubemap_water_fs_code);
  GLuint skybox_shader_program = create_shader_program(skybox_vs_code, skybox_fs_code);

  Texture2D white_texture;
  float movement_speed = 1.0;
  float rotation_speed = 30.0;
  Camera camera(glm::vec3(0.0, 1.0, 1.0),
    0.0, 0.0, 45.0f, 1260.0f / 1080.0f, 0.01, 1000.0, 
    rotation_speed, movement_speed
  );

  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


  // Skybox Shader Uniforms
  GLuint skybox_view_proj_loc = glGetUniformLocation(skybox_shader_program, "u_ViewProjection");
  GLuint skybox_texture_loc = glGetUniformLocation(skybox_shader_program, "cube_map");
  glUseProgram(skybox_shader_program);
  glUniform1f(skybox_texture_loc, 0);

  // Textures received from: https://www.humus.name/index.php?page=Textures
  const char* skyboxes_names[] = { "skansen", "ocean", "church"};
  static int current_skybox_idx = 0;
  const char* skybox_combo_label = skyboxes_names[current_skybox_idx];
  Skybox skyboxes[] = {Skybox(skyboxes_names[0], true), Skybox(skyboxes_names[1], true), Skybox(skyboxes_names[2], true)};

  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
  OceanSettings ocean_settings;
  Ocean ocean(ocean_settings);

  bool draw_skybox = true;
  bool wire_frame = false;
  bool use_skybox_shader = true;

  int n = 0;
  std::vector<double> times(100);
  std::vector<double> fps(100);
  while (!window.should_close ()) {
    glClearColor(0.8, 0.85, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
    if (wire_frame)
      glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    else
      glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

    double dt = clock.tick();
    update(window, dt, camera);
    double start = clock.since_start();
    ocean.update(dt);
    times[n % 100] = (clock.since_start() - start) * 1000;
    fps[n % 100] = dt;
    n++;

    if (use_skybox_shader)
      ocean.draw(cubemap_water_shader, skyboxes[current_skybox_idx], camera);
    else
      ocean.draw(color_water_shader, skyboxes[current_skybox_idx], camera);

    /** SKYBOX RENDERING BEGIN (done last) **/
    if (draw_skybox) {
      glm::mat4 skybox_view_projection = camera.get_view_projection(false);
      glUseProgram(skybox_shader_program);
      glUniformMatrix4fv(skybox_view_proj_loc, 1, false, &skybox_view_projection[0][0]);
      skyboxes[current_skybox_idx].draw();
    }
    /** SKYBOX RENDERING END **/

    double update_sum = 0.0;
    double fps_sum = 0.0;
    for (int i = 0; i < 100; i++) {
      update_sum += times[i];
      fps_sum += fps[i];
    }
    update_sum /= 100;
    fps_sum /= 100;
      

    /** GUI RENDERING BEGIN **/
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::Begin("Settings panel");
    ImGui::Text("FPS: %f", (1 / fps_sum));
    ImGui::Text("Update-time: %f (ms)", (update_sum));
    ImGui::Text("Simulation");
    ImGui::Dummy(ImVec2(0.0, 5.0));
    //ImGui::SliderFloat("Simulation speed", &simulation_speed, 0.0, 10.0);
    //ImGui::SliderInt("Num tiles", &num_tiles, 1, 20);
    ImGui::Checkbox("Wireframe", &wire_frame);
    ImGui::Dummy(ImVec2(0.0, 15.0));

    ImGui::Text("Wave");
    ImGui::Dummy(ImVec2(0.0, 5.0));
    ImGui::Text("Real-time parameters");
    ImGui::SliderInt("Tile count", &ocean.num_tiles, 1, 20);
    ImGui::SliderFloat("Vertex Distance", &ocean.vertex_distance, 1.0, 20.0);
    ImGui::SliderFloat("Simulation speed", &ocean.simulation_speed, 0.0, 10.0);
    ImGui::SliderFloat("Normal roughness", &ocean.normal_roughness, 1.0, 20.0);
    ImGui::Dummy(ImVec2(0.0, 5.0));
    ImGui::Text("Reloadable parameters");
    // TODO: Fix combobox with N;
    ImGui::SliderFloat("Length", &ocean_settings.length, 0.0f, 1000.0f);
    ImGui::SliderFloat("Amplitude", &ocean_settings.amplitude, 0.0f, 20.0f);
    ImGui::SliderFloat("Wind speed", &ocean_settings.wind_speed, 0.0f, 200.0f);
    if (ImGui::Button("Reload"))
      ocean.reload_settings(ocean_settings);
    ImGui::Dummy(ImVec2(0.0, 15.0));

    ImGui::Text("Environment");
    ImGui::Dummy(ImVec2(0.0, 5.0));
    ImGui::Checkbox("Skybox shader", &use_skybox_shader);
    ImGui::Checkbox("Enable skybox", &draw_skybox);
    
    if (ImGui::BeginCombo("Skybox", skybox_combo_label))
    {
      for (int n = 0; n < IM_ARRAYSIZE(skyboxes_names); n++)
      {
          const bool is_selected = (current_skybox_idx == n);
          if (ImGui::Selectable(skyboxes_names[n], is_selected)) current_skybox_idx = n;
          if (is_selected) ImGui::SetItemDefaultFocus();
      }
      ImGui::EndCombo();
    }
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

