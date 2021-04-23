#include "wave_utils.h"

#include <cmath>
#include <random>

static std::default_random_engine generator;
static std::normal_distribution<double> distribution(0.0, 1.0);

const double g = 9.82;

/**
* "A useful model for wind-driven waves larger than
* capillary waves in a fully developed sea is the Phillips spectrum"
*
* Equation 40 in Tessendorf (2001)
*/
double phillips(const glm::vec2& k) {
    // Constants TODO: Move to alterable place
    double amplitude = 4.f;
    double wind_speed = 31.f;
    glm::vec2 wind_dir(1.0, 0.0);

    double L = wind_speed * wind_speed / g;
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
    return sqrt(glm::length(K) * 9.82);
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
