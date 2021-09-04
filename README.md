# Water rendering
This project aims to create a scene for rendering realistically looking water in real time based on Jerry Tessendorf's paper Simulating Ocean Water (2001). 

The project uses CUDA ([cuFFT](https://developer.nvidia.com/cufft)) to accelerate fourier transforms to generate the wave animation.

![Example scene](/images/short.gif)

*Example scene rendered using the application*

## Dependencies
Following dependencies are required to run the application.

### Submodules
- glad
- glfw
- glm
- imgui

### External
- [CMake](https://cmake.org/download/)
- [CUDA](https://developer.nvidia.com/cuda-downloads)

Due to CUDA dependency, application is currently dependent on NVIDIA hardware with updated drivers. 

## Build
```
git clone --recurse-submodules https://github.com/raaavioli/WaterRendering
cd WaterRendering
mkdir build && cd build
cmake ..
make
```

Building project creates binary **WaterRendering**.

## Development
Development process tracked on my [blog](https://ocean-water-simulation.blogspot.com).

## References
1. Jerry Tessendorf, "Simulating Ocean Water" Simulating Nature: Realistic and Interactive Techniques Course
Notes, SIGGRAPH 1999. [doi](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.161.9102&rep=rep1&type=pdf)
2. Fredrik Larsson, "Deterministic Ocean Waves", Master's thesis, Lund University 2012. [doi](https://sam.cs.lth.se/ExjobGetFile?id=514)
