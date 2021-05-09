#include "ocean.h"

#include <cmath>
#include <random>

#include <glm/gtx/transform.hpp>

static std::default_random_engine generator;
static std::normal_distribution<double> distribution(0.0, 1.0);

Ocean::Ocean(OceanSettings settings) : settings(settings) {
    std::vector<uint32_t> indices;
    int N = settings.N;
    int Nplus1 = N + 1;

    vertices.resize(Nplus1 * Nplus1);
    indices.reserve(N * N * 6);
    for (int z = 0; z < Nplus1; z++) {
        for (int x = 0; x < Nplus1; x++) {
            int i0 = z * Nplus1 + x;
            Vertex vertex;
            vertex.color = glm::normalize(glm::vec3(29,162,216));
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
    reload_settings(settings);
}

Ocean::~Ocean() {
    if (allocation_host) {
        delete[] allocation_host;
        allocation_host = nullptr;
    };
    CUDA_ASSERT(cudaFree(allocation_device));
}

void Ocean::update(double dt) {
    simulation_time += simulation_speed * dt;
    float length = settings.length;
    int N = settings.N;

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

    // Copy displacement_y, displacement_x, displacement_z, gradient_x, gradient_z simultaneously to device
    CUDA_ASSERT(cudaMemcpy(displacement_y_device, displacement_y, 5 * sizeof(std::complex<double>) * N * N, cudaMemcpyHostToDevice));
    // In place transforms
    // Inverse FFT: h(k, x, t) = sum(h_tilde(k, x, t) * e^(2pikn / N))
    CUFFT_ASSERT(cufftExecZ2Z(plan, displacement_y_device, displacement_y_device, CUFFT_INVERSE)); 
    CUFFT_ASSERT(cufftExecZ2Z(plan, displacement_x_device, displacement_x_device, CUFFT_INVERSE)); 
    CUFFT_ASSERT(cufftExecZ2Z(plan, displacement_z_device, displacement_z_device, CUFFT_INVERSE)); 
    CUFFT_ASSERT(cufftExecZ2Z(plan, gradient_x_device, gradient_x_device, CUFFT_INVERSE)); 
    CUFFT_ASSERT(cufftExecZ2Z(plan, gradient_z_device, gradient_z_device, CUFFT_INVERSE)); 

    // Copy displacement_y, displacement_x, displacement_z, gradient_x, gradient_z simultaneously from device
    CUDA_ASSERT(cudaMemcpy(displacement_y, displacement_y_device, 5 * sizeof(std::complex<double>) * N * N, cudaMemcpyDeviceToHost));

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

void Ocean::draw(uint32_t shader, const Skybox& skybox, const Camera& camera) {
    GLuint shader_view_proj_loc = glGetUniformLocation(shader, "u_ViewProjection");
    GLuint model_loc = glGetUniformLocation(shader, "u_Model");
    GLuint camera_pos_loc = glGetUniformLocation(shader, "u_CameraPos");
    GLuint shader_cube_map_loc  = glGetUniformLocation(shader, "cube_map");

    glm::vec3 camera_position = camera.get_position();
    glm::mat4 wave_view_projection = camera.get_view_projection(true);
    glm::mat4 water_matrix = glm::identity<glm::mat4>();
    water_matrix = glm::rotate(water_matrix, glm::radians<float>(90), glm::vec3(1.0, 0.0, 0.0));

    glUseProgram(shader);
    glUniformMatrix4fv(shader_view_proj_loc, 1, false, &wave_view_projection[0][0]);
    glUniformMatrix4fv(model_loc, 1, false, &water_matrix[0][0]);
    glUniform3f(camera_pos_loc, camera_position.x, camera_position.y, camera_position.z);
    glUniform1i(shader_cube_map_loc, 0);


    this->surface_model.bind();
    skybox.bind_cube_map(0);
    for (int z = 0; z < num_tiles; z++) {
      for (int x = 0; x < num_tiles; x++) {
        glm::mat4 water_matrix = glm::translate(glm::mat4(1.0), 
          glm::vec3(vertex_distance * (-num_tiles / 2.0f + x), 0.0, -3.0 + vertex_distance * (-num_tiles / 2.0f + z))
        );
        glUniformMatrix4fv(model_loc, 1, false, &water_matrix[0][0]);
        this->surface_model.draw();
      }
    }
    skybox.unbind_cube_map();
}

void Ocean::reload_settings(OceanSettings new_settings) {
    if (allocation_host)
        delete[] allocation_host;
    settings = new_settings;
    int N = new_settings.N;
    float length = new_settings.length;
    
    allocation_host = new std::complex<double>[7 * N * N];
    h0_tk = allocation_host + 0 * N * N; // h0_tilde(k)
    h0_tmk = allocation_host + 1 * N * N; // h0_tilde(-k)

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

    displacement_y = allocation_host + 2 * N * N; // h(k, x, t)
    displacement_x = allocation_host + 3 * N * N; // x-displacement of h(k, x, t)
    displacement_z = allocation_host + 4 * N * N; // z-displacement of h(k, x, t)
    gradient_x = allocation_host + 5 * N * N; // x-gradient of h(k, x, t)
    gradient_z = allocation_host + 6 * N * N; // z-gradient of h(k, x, t)

    if (allocation_device)
        CUDA_ASSERT(cudaFree(allocation_device));
    CUDA_ASSERT(cudaMalloc ((void**) &allocation_device, 5 * sizeof(std::complex<double>) * N * N));
    displacement_y_device = allocation_device + 0 * N * N;
    displacement_x_device = allocation_device + 1 * N * N;
    displacement_z_device = allocation_device + 2 * N * N;
    gradient_x_device = allocation_device + 3 * N * N;
    gradient_z_device = allocation_device + 4 * N * N;

    CUFFT_ASSERT(cufftPlan2d(&plan, N, N, CUFFT_Z2Z));

    std::vector<uint32_t> indices;
    indices.reserve(N * N * 6);
    int Nplus1 = N + 1;
    for (int z = 0; z < Nplus1; z++) {
        for (int x = 0; x < Nplus1; x++) {
            if (x < N && z < N) {
                int i0 = z * Nplus1 + x;
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
    this->surface_model.update_index_data(indices);
}

void Ocean::update_vertices() {
    int N = settings.N;
    int Nplus1 = N + 1;
    for (int z = 0; z < Nplus1; z++) {
        for (int x = 0; x < Nplus1; x++) {
            int i_v = z * Nplus1 + x;
            int i_d = (z % N) * N + x % N;

            glm::vec3 origin_position = glm::vec3(-0.5 + x * vertex_distance / float(N), 0, -0.5 + z * vertex_distance / float(N));
            glm::vec3 displacement(
                choppiness * displacement_x[i_d].real(), 
                displacement_y[i_d].real(), 
                choppiness * displacement_z[i_d].real()
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
            vertices[i_v].normal = glm::vec3(-ex * normal_roughness, 1.0, -ez * normal_roughness);
        }
    }
}

/**
* "A useful model for wind-driven waves larger than
* capillary waves in a fully developed sea is the Phillips spectrum"
*
* Equation 40 in Tessendorf (2001)
*/
double Ocean::phillips(const glm::vec2& k) {
    double L = settings.wind_speed * settings.wind_speed / Ocean::g;
    double k_len = glm::length(k);
    k_len = (k_len < 0.0001) ? 0.0001 : k_len; // to avoid divide by 0
    double k2 = k_len * k_len;
    double k4 = k2 * k2;

    double kw = 0.0;
    if (k.x || k.y) {
        kw = glm::dot(glm::normalize(k), glm::normalize(settings.wind_dir));
    }

    double res = settings.amplitude * kw * kw * exp(-1 / (k2 * L * L)) / k4;

    return res;
}

/**
* Dispersion relation suggested with regard to depth d: 
*   sqrt(k * g * tanh(k * d))
* Notice: for large d, tanh = 1, so formula equals.
*   sqrt(k * g)
*/
double Ocean::dispersion(const glm::vec2& K) {
    return sqrt(glm::length(K) * Ocean::g);
}

/**
* Equation 42 in Tessendorf (2001)
*/
std::complex<double> Ocean::h0_tilde(const glm::vec2& K) {
    double er = distribution(generator);
    double ei = distribution(generator);

    return sqrt(phillips(K)) * (std::complex(er, ei)) / sqrt(2.0);
}

/**
* Equation 43 in Tessendorf (2001)
*/
std::complex<double> Ocean::h_tilde(const std::complex<double>& h0_tk, const std::complex<double>& h0_tmk, const glm::vec2& K, double t) {
    double wkt = dispersion(K) * t;
    return h0_tk * exp(std::complex(0.0, wkt)) + std::conj(h0_tmk) * exp(std::complex(0.0, -wkt));
}