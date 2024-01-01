#ifndef RAY_H
#define RAY_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include "vec3.cuh"

class ray {
  public:
    CUDA_CALLABLE_MEMBER ray() {}

    CUDA_CALLABLE_MEMBER ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}

    CUDA_CALLABLE_MEMBER point3 origin() const  { return orig; }
    CUDA_CALLABLE_MEMBER vec3 direction() const { return dir; }

    CUDA_CALLABLE_MEMBER point3 at(double t) const {
        return orig + t*dir;
    }

  private:
    point3 orig;
    vec3 dir;
};

#endif