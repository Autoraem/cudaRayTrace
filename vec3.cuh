#ifndef VEC3_H
#define VEC3_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include <cuda_runtime.h>
#include <iostream>

using std::sqrt;

class vec3 {
public:
    double e[3];

    CUDA_CALLABLE_MEMBER vec3() : e{0,0,0} {}
    CUDA_CALLABLE_MEMBER vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

    CUDA_CALLABLE_MEMBER double x() const { return e[0]; }
    CUDA_CALLABLE_MEMBER double y() const { return e[1]; }
    CUDA_CALLABLE_MEMBER double z() const { return e[2]; }

    CUDA_CALLABLE_MEMBER vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    CUDA_CALLABLE_MEMBER double operator[](int i) const { return e[i]; }
    CUDA_CALLABLE_MEMBER double& operator[](int i) { return e[i]; }

    CUDA_CALLABLE_MEMBER vec3& operator+=(const vec3 &v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    CUDA_CALLABLE_MEMBER vec3& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    CUDA_CALLABLE_MEMBER vec3& operator/=(double t) {
        return *this *= 1/t;
    }

    CUDA_CALLABLE_MEMBER double length() const {
        return sqrtf(length_squared());
    }

    CUDA_CALLABLE_MEMBER double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;


// Vector Utility Functions

inline CUDA_CALLABLE_MEMBER std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline CUDA_CALLABLE_MEMBER vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline CUDA_CALLABLE_MEMBER vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline CUDA_CALLABLE_MEMBER vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline CUDA_CALLABLE_MEMBER vec3 operator*(double t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline CUDA_CALLABLE_MEMBER vec3 operator*(const vec3 &v, double t) {
    return t * v;
}

inline CUDA_CALLABLE_MEMBER vec3 operator/(vec3 v, double t) {
    return (1/t) * v;
}

inline CUDA_CALLABLE_MEMBER double dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

inline CUDA_CALLABLE_MEMBER vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline CUDA_CALLABLE_MEMBER vec3 unit_vector(vec3 v) {
    return v / v.length();
}

#endif
