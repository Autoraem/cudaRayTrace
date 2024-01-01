// Responsible for (1) constructing and dispatching rays into the worlds
// (2) Using the results of these rays to construct the rendered image
// Taken from "Ray Tracing in One Weekend"

#ifndef CAMERA_H
#define CAMERA_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include "ray.cuh"
#include "vec3.cuh"
#include "color.cuh"

//Function prototypes
__global__ 
void renderKernel(int image_width, int image_height, vec3 pixel00_loc, vec3 pixel_delta_u, vec3 pixel_delta_v, vec3 camera_center, double *output);

class Camera {
    public:
        double aspect_ratio;
        int image_width;
        int image_height;

        double focal_length;
        double viewport_height;
        double viewport_width;
        vec3 viewport_u;
        vec3 viewport_v;
        vec3 viewport_upper_left;
        vec3 camera_center;

        vec3 pixel_delta_u;
        vec3 pixel_delta_v;
        vec3 pixel00_loc;

        double *h_output;
        double *d_output;

    Camera(){
        aspect_ratio = 1;
        image_width = 400;
        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        focal_length = 1.0;
        viewport_height = 2.0;
        viewport_width = aspect_ratio * viewport_height;
        viewport_u = vec3(viewport_width, 0, 0);
        viewport_v = vec3(0, -viewport_height, 0);
        viewport_upper_left = vec3(-viewport_width/2, viewport_height/2, -focal_length);

        camera_center = vec3(0, 0, 0);
        pixel_delta_u = viewport_u / (image_width - 1);
        pixel_delta_v = -viewport_v / (image_width - 1);
        pixel00_loc = viewport_upper_left + (pixel_delta_u / 2) + (pixel_delta_v / 2);

        // Allocate host memory
        size_t fb_size = 3 * image_width * image_height * sizeof(double);
        h_output = new double[fb_size];

        // Allocate device memory
        cudaMalloc((void**)&d_output, fb_size);
    }

    void deleteCamera() {
        // Free device memory
        cudaFree(d_output);

        // Free host memory
        delete[] h_output;
    }

    void render();

    private:
        __device__
        color ray_color(const ray& r);

};

void Camera::render() {
    // Launch kernel
    int tx = 8;
    int ty = 8;
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threadsPerBlock(tx, ty);
    renderKernel<<<blocks, threadsPerBlock>>>(image_width, image_height, pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center, d_output);
    //Sync device
    cudaDeviceSynchronize();
    // Copy results from device to host
    cudaMemcpy(h_output, d_output, image_width * image_height * sizeof(vec3), cudaMemcpyDeviceToHost);
    

}

__device__
color ray_color(const ray& r) {
    //this blend function is a linear blend between white and blue depending on the y coordinate of the ray
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y() + 1.0);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    // return color(1, 0, 0);
}

// CUDA kernel function
__global__ 
void renderKernel(int image_width, int image_height, vec3 pixel00_loc, vec3 pixel_delta_u, vec3 pixel_delta_v, vec3 camera_center, double *output){
    // i and j are the pixel coordinates corresponding to the thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    //if threads are in the image, set the pixel color
    if (i < image_width && j < image_height) {
        int pixel_index = j * image_width + i;

        auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        auto ray_direction = pixel_center - camera_center;
        ray r(camera_center, ray_direction);

        color pixel_color = ray_color(r);

        output[pixel_index + 0] = pixel_color.x();
        output[pixel_index + 1] = pixel_color.y();
        output[pixel_index + 2] = pixel_color.z();
    }
}



#endif

