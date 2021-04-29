#ifndef WR_OCEAN_H
#define WR_OCEAN_H

#include <complex>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cufft.h>

#define CUDA_ASSERT(err) if(err != cudaSuccess) std::cout << "Cuda Error: " << err << ", Line: " << __LINE__ << std::endl;
#define CUFFT_ASSERT(err) if(err != CUFFT_SUCCESS) std::cout << "Cufft Error: " << err << ", Line: " << __LINE__ << std::endl;

#include <glm/glm.hpp>
#include "model.h"
#include "camera.h"

double phillips(const glm::vec2& K);
double dispersion(const glm::vec2& K);
std::complex<double> h0_tilde(const glm::vec2& K);
std::complex<double> h_tilde(const std::complex<double>& h0_tk, const std::complex<double>& h0_tmk, const glm::vec2& K, double t);

struct Ocean {
    Ocean(int N, Skybox skybox);
    ~Ocean();

    void update(double dt);
    void draw(uint32_t shader, const Camera& camera);

    static constexpr double g = 9.82;

private:
    void update_vertices();

private:
    int N;
    int Nplus1;
    float tile_dim = 5.0;
    double simulation_time = 0.0;
    double length = 300;
    double two_pi = glm::two_pi<double>();
    
    int num_tiles = 10;
    float simulation_speed = 2.0;

    RawModel surface_model;
    Skybox skybox;

    std::vector<Vertex> vertices;

    std::complex<double>* h0_tk; // h0_tilde(k)
    std::complex<double>* h0_tmk; // h0_tilde(-k)

    std::complex<double>* displacement_y; // h~(k, x, t) -> h(k, x, t)
    std::complex<double>* displacement_x; // x-displacement of h(k, x, t)
    std::complex<double>* displacement_z; // z-displacement of h(k, x, t)
    std::complex<double>* gradient_x; // x-gradient of h(k, x, t)
    std::complex<double>* gradient_z; // z-gradient of h(k, x, t)

    cufftDoubleComplex* displacement_y_device;
    cufftDoubleComplex* displacement_x_device;
    cufftDoubleComplex* displacement_z_device;
    cufftDoubleComplex* gradient_x_device;
    cufftDoubleComplex* gradient_z_device;

    cufftHandle plan;
};

#endif //WR_OCEAN_H