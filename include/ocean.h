#ifndef WR_OCEAN_H
#define WR_OCEAN_H

#include <complex>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cufft.h>

#define CUDA_ASSERT(err) if(err != cudaSuccess) std::cout \
    << "Cuda Error: " << err << ", Line: " << __LINE__ << ", " \
    << cudaGetErrorString(err) << std::endl;
#define CUFFT_ASSERT(err) if(err != CUFFT_SUCCESS) std::cout << "Cufft Error: " << err << ", Line: " << __LINE__ << std::endl;

#include <glm/glm.hpp>

#include "model.h"
#include "camera.h"

struct OceanSettings {
    int N = 256;
    float length = N * 0.75;
    float amplitude = 5.f;
    float wind_speed = 32.f;
    glm::vec2 wind_dir = glm::vec2(1.0, 0.0);
};

struct Ocean {
    Ocean(OceanSettings settings);
    ~Ocean();

    void update(double dt);
    void draw(uint32_t shader, const Skybox& skybox, const Camera& camera);
    void reload_settings(OceanSettings new_settings);

    static constexpr double g = 9.82;

public:
    // Real-time parameters
    int num_tiles = 10;
    float vertex_distance = 5.0;
    float simulation_speed = 2.0;
    float normal_roughness = 5.0;
    float choppiness = -1.0;

private:
    void update_vertices();
    double phillips(const glm::vec2& K);
    double dispersion(const glm::vec2& K);
    std::complex<double> h0_tilde(const glm::vec2& K);
    std::complex<double> h_tilde(const std::complex<double>& h0_tk, 
        const std::complex<double>& h0_tmk, const glm::vec2& K, double t);

private:
    const double two_pi = glm::two_pi<double>();
    double simulation_time = 0.0;
    OceanSettings settings;

    RawModel surface_model;

    std::vector<Vertex> vertices;

    /* Memory */
    // Base allocation for host data
    std::complex<double>* allocation_host = nullptr;

    // Pointers into allocation_host
    std::complex<double>* h0_tk; // h0_tilde(k)
    std::complex<double>* h0_tmk; // h0_tilde(-k)
    std::complex<double>* displacement_y; // h~(k, x, t) -> h(k, x, t)
    std::complex<double>* displacement_x; // x-displacement of h(k, x, t)
    std::complex<double>* displacement_z; // z-displacement of h(k, x, t)
    std::complex<double>* gradient_x; // x-gradient of h(k, x, t)
    std::complex<double>* gradient_z; // z-gradient of h(k, x, t)

    // Base allocation for device data
    cufftDoubleComplex* allocation_device = nullptr;

    // Pointers into allocation_device
    cufftDoubleComplex* displacement_y_device;
    cufftDoubleComplex* displacement_x_device;
    cufftDoubleComplex* displacement_z_device;
    cufftDoubleComplex* gradient_x_device;
    cufftDoubleComplex* gradient_z_device;

    cufftHandle plan;
};

#endif //WR_OCEAN_H