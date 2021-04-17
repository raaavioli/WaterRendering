#ifndef WAVE_UTILS_H
#define WAVE_UTILS_H

#include <complex>

#include <glm/glm.hpp>

double phillips(const glm::vec2& K);

double dispersion(const glm::vec2& K);

std::complex<double> h0_tilde(const glm::vec2& K);

std::complex<double> h_tilde(const std::complex<double>& h0_tk, const std::complex<double>& h0_tmk, const glm::vec2& K, double t);

#endif // WAVE_UTILS_H