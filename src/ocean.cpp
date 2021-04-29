#include "ocean.h"

#include <cmath>
#include <random>

#include <glm/gtx/transform.hpp>

static std::default_random_engine generator;
static std::normal_distribution<double> distribution(0.0, 1.0);

/**
* "A useful model for wind-driven waves larger than
* capillary waves in a fully developed sea is the Phillips spectrum"
*
* Equation 40 in Tessendorf (2001)
*/
double phillips(const glm::vec2& k) {
    // Constants TODO: Move to alterable place
    double amplitude = 4.f;
    double wind_speed = 15.f;
    glm::vec2 wind_dir(1.0, 0.0);

    double L = wind_speed * wind_speed / Ocean::g;
    double k_len = glm::length(k);
    k_len = (k_len < 0.0001) ? 0.0001 : k_len; // to avoid divide by 0
    double k2 = k_len * k_len;
    double k4 = k2 * k2;

    double kw = 0.0;
    if (k.x || k.y) {
        kw = glm::dot(glm::normalize(k), glm::normalize(wind_dir));
    }

    double res = amplitude * kw * kw * exp(-1 / (k2 * L * L)) / k4;

    return res;
}

/**
* Dispersion relation suggested with regard to depth d: 
*   sqrt(k * g * tanh(k * d))
* Notice: for large d, tanh = 1, so formula equals.
*   sqrt(k * g)
*/
double dispersion(const glm::vec2& K) {
    return sqrt(glm::length(K) * Ocean::g);
}

/**
* Equation 42 in Tessendorf (2001)
*/
std::complex<double> h0_tilde(const glm::vec2& K) {
    double er = distribution(generator);
    double ei = distribution(generator);

    return sqrt(phillips(K)) * (std::complex(er, ei)) / sqrt(2.0);
}

/**
* Equation 43 in Tessendorf (2001)
*/
std::complex<double> h_tilde(const std::complex<double>& h0_tk, const std::complex<double>& h0_tmk, const glm::vec2& K, double t) {
    double wkt = dispersion(K) * t;
    return h0_tk * exp(std::complex(0.0, wkt)) + std::conj(h0_tmk) * exp(std::complex(0.0, -wkt));
}

Ocean::Ocean(int N, Skybox skybox) : N(N), Nplus1(N + 1), skybox(skybox), vertices(Nplus1 * Nplus1) {
    std::vector<uint32_t> indices;
    indices.reserve(N * N * 6);
    for (int z = 0; z < Nplus1; z++) {
        for (int x = 0; x < Nplus1; x++) {
            int i0 = z * Nplus1 + x;
            Vertex vertex;
            vertex.color = glm::vec3(1.0, 0.0, 0.0);
            vertices[i0] = vertex;

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

  surface_model = RawModel(vertices, indices, GL_DYNAMIC_DRAW);

  h0_tk = new std::complex<double>[N * N]; // h0_tilde(k)
  h0_tmk = new std::complex<double>[N * N]; // h0_tilde(-k)

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

  displacement_y = new std::complex<double>[N * N]; // h(k, x, t)
  displacement_x = new std::complex<double>[N * N]; // x-displacement of h(k, x, t)
  displacement_z = new std::complex<double>[N * N]; // z-displacement of h(k, x, t)
  gradient_x = new std::complex<double>[N * N]; // x-gradient of h(k, x, t)
  gradient_z = new std::complex<double>[N * N]; // z-gradient of h(k, x, t)

  CUDA_ASSERT(cudaMalloc ((void**) &displacement_y_device, sizeof(std::complex<double>) * N * N));
  CUDA_ASSERT(cudaMalloc ((void**) &displacement_x_device, sizeof(std::complex<double>) * N * N));
  CUDA_ASSERT(cudaMalloc ((void**) &displacement_z_device, sizeof(std::complex<double>) * N * N));
  CUDA_ASSERT(cudaMalloc ((void**) &gradient_x_device, sizeof(std::complex<double>) * N * N));
  CUDA_ASSERT(cudaMalloc ((void**) &gradient_z_device, sizeof(std::complex<double>) * N * N));

  CUFFT_ASSERT(cufftPlan2d(&plan, N, N, CUFFT_Z2Z));
}

Ocean::~Ocean() {
    delete[] h0_tk; // h0_tilde(k)
    delete[] h0_tmk; // h0_tilde(-k)

    delete[] displacement_y; // h(k, x, t)
    delete[] displacement_x; // x-displacement of h(k, x, t)
    delete[] displacement_z; // z-displacement of h(k, x, t)
    delete[] gradient_x; // x-gradient of h(k, x, t)
    delete[] gradient_z; // z-gradient of h(k, x, t)
}

void Ocean::update(double dt) {
    simulation_time += simulation_speed * dt;
    // Setup h_tk + device
    for (int m = 0; m < N; m++) {
      for (int n = 0; n < N; n++) {
        int i = m * N + n;
        float kx = (n - N / 2.f) * two_pi / length;
        float kz = (m - N / 2.f) * two_pi / length;
        glm::vec2 K(kx, kz);
        displacement_y[i] = h_tilde(h0_tk[i], h0_tmk[i], K, simulation_time); // Initial displacement h_tilde(k, x, t)
        gradient_x[i] = displacement_y[i] * std::complex<double>(0.0, kx);
        gradient_z[i] = displacement_y[i] * std::complex<double>(0.0, kz);
        double k_length = glm::length(K);
        if (k_length > 0.00001) {
          displacement_x[i] = displacement_y[i] * std::complex<double>(0.0, -kx / k_length);
          displacement_z[i] = displacement_y[i] * std::complex<double>(0.0, -kz / k_length);
        } else {
          displacement_x[i] = displacement_y[i] * std::complex<double>(0.0, 0.0);
          displacement_z[i] = displacement_y[i] * std::complex<double>(0.0, 0.0);
        }
      }
    }

    CUDA_ASSERT(cudaMemcpy(displacement_y_device, displacement_y, sizeof(std::complex<double>) * N * N, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(displacement_x_device, displacement_x, sizeof(std::complex<double>) * N * N, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(displacement_z_device, displacement_z, sizeof(std::complex<double>) * N * N, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(gradient_x_device, gradient_x, sizeof(std::complex<double>) * N * N, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(gradient_z_device, gradient_z, sizeof(std::complex<double>) * N * N, cudaMemcpyHostToDevice));
    // In place transforms
    // Inverse FFT: h(k, x, t) = sum(h_tilde(k, x, t) * e^(2pikn / N))
    CUFFT_ASSERT(cufftExecZ2Z(plan, displacement_y_device, displacement_y_device, CUFFT_INVERSE)); 
    CUFFT_ASSERT(cufftExecZ2Z(plan, displacement_x_device, displacement_x_device, CUFFT_INVERSE)); 
    CUFFT_ASSERT(cufftExecZ2Z(plan, displacement_z_device, displacement_z_device, CUFFT_INVERSE)); 
    CUFFT_ASSERT(cufftExecZ2Z(plan, gradient_x_device, gradient_x_device, CUFFT_INVERSE)); 
    CUFFT_ASSERT(cufftExecZ2Z(plan, gradient_z_device, gradient_z_device, CUFFT_INVERSE)); 

    CUDA_ASSERT(cudaMemcpy(displacement_y, displacement_y_device, sizeof(std::complex<double>) * N * N, cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(displacement_x, displacement_x_device, sizeof(std::complex<double>) * N * N, cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(displacement_z, displacement_z_device, sizeof(std::complex<double>) * N * N, cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(gradient_x, gradient_x_device, sizeof(std::complex<double>) * N * N, cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(gradient_z, gradient_z_device, sizeof(std::complex<double>) * N * N, cudaMemcpyDeviceToHost));

    for (int m = 0; m < N; m++) {
      for (int n = 0; n < N; n++) {
        int i = m * N + n;
        int sign = (m + n) % 2 == 0 ? 1 : -1; // Larsson (2012), Equation 4.6
        displacement_y[i] /= sign * (N * N);
        displacement_x[i] /= sign * (N * N);
        displacement_z[i] /= sign * (N * N);
        gradient_x[i] /= sign * (N * N);
        gradient_z[i] /= sign * (N * N);
      }
    }

    update_vertices();
    this->surface_model.update_vertex_data(vertices);
}

void Ocean::draw(uint32_t shader, const Camera& camera) {
    GLuint shader_view_proj_loc = glGetUniformLocation(shader, "u_ViewProjection");
    GLuint model_loc = glGetUniformLocation(shader, "u_Model");
    GLuint time_loc = glGetUniformLocation(shader, "u_Time");
    GLuint camera_pos_loc = glGetUniformLocation(shader, "u_CameraPos");
    GLuint tex0_loc = glGetUniformLocation(shader, "texture0");
    GLuint shader_cube_map_loc  = glGetUniformLocation(shader, "cube_map");

    glm::vec3 camera_position = camera.get_position();
    glm::mat4 wave_view_projection = camera.get_view_projection(true);
    glm::mat4 water_matrix = glm::identity<glm::mat4>();
    water_matrix = glm::rotate(water_matrix, glm::radians<float>(90), glm::vec3(1.0, 0.0, 0.0));

    glUseProgram(shader);
    glUniformMatrix4fv(shader_view_proj_loc, 1, false, &wave_view_projection[0][0]);
    glUniformMatrix4fv(model_loc, 1, false, &water_matrix[0][0]);
    glUniform1f(time_loc, simulation_time);
    glUniform3f(camera_pos_loc, camera_position.x, camera_position.y, camera_position.z);
    glUniform1i(tex0_loc, 0);
    glUniform1i(shader_cube_map_loc, 1);


    this->surface_model.bind();
    this->skybox.bind_cube_map(1);
    //desert_skybox.bind_cube_map(1);
    for (int z = 0; z < num_tiles; z++) {
      for (int x = 0; x < num_tiles; x++) {
        glm::mat4 water_matrix = glm::translate(glm::mat4(1.0), 
          glm::vec3(tile_dim * (-num_tiles / 2.0f + x), 0.0, -3.0 + tile_dim * (-num_tiles / 2.0f + z))
        );
        glUniformMatrix4fv(model_loc, 1, false, &water_matrix[0][0]);
        this->surface_model.draw();
      }
    }

    this->skybox.unbind_cube_map();
}

void Ocean::update_vertices() {
    int N = Nplus1 - 1;
    for (int z = 0; z < Nplus1; z++) {
        for (int x = 0; x < Nplus1; x++) {
            int i_v = z * Nplus1 + x;
            int i_d = (z % N) * N + x % N;
            double lambda = -1.0;

            glm::vec3 origin_position = glm::vec3(-0.5 + x * tile_dim / float(N), 0, -0.5 + z * tile_dim / float(N));
            glm::vec3 displacement(
                lambda * displacement_x[i_d].real(), 
                displacement_y[i_d].real(), 
                lambda * displacement_z[i_d].real()
            );
            vertices[i_v].position = origin_position + displacement;
        }
    }

    for (int z = 0; z < Nplus1; z++) {
        for (int x = 0; x < Nplus1; x++) {
            int i_v = z * Nplus1 + x;
            int i_d = (z % N) * N + x % N;
            double ex = gradient_x[i_d].real();
            double ez = gradient_z[i_d].real();
            vertices[i_v].normal = glm::vec3(-ex * 5.0, 1.0, -ez * 5.0);
        }
    }
}