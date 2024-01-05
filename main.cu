#include "color.cuh"
#include "vec3.cuh"
#include "ray.cuh"
#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// // __device__
// // color ray_color(const ray& r) {
// //     //this blend function is a linear blend between white and blue depending on the y coordinate of the ray
// //     vec3 unit_direction = unit_vector(r.direction());
// //     auto a = 0.5*(unit_direction.y() + 1.0);
// //     return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
// //     // return color(1, 0, 0);
// // }

// // // CUDA kernel function
// __global__ 
// void renderKernel(int image_width, int image_height, double *output){
//     // i and j are the pixel coordinates corresponding to the thread index
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     int pixel_index = j * image_width + i;
    
//     //if threads are in the image, set the pixel color
//     if (i < image_width && j < image_height) {
        
//         auto r = double(i) / (image_width-1);
//         auto g = double(j) / (image_height-1);
//         auto b = 0;

//         output[pixel_index + 0] = r;
//         output[pixel_index + 1] = g;
//         output[pixel_index + 2] = b;
//     }
// }

// // int main() {
// //     double aspect_ratio = 16.0/9.0;
// //     int image_width = 256;
// //     int image_height = static_cast<int>(image_width / aspect_ratio);
// //     image_height = (image_height < 1) ? 1 : image_height;

// //     // double focal_length = 1.0;
// //     // double viewport_height = 2.0;
// //     // double viewport_width = aspect_ratio * viewport_height;
// //     // vec3 viewport_u = vec3(viewport_width, 0, 0);
// //     // vec3 viewport_v = vec3(0, -viewport_height, 0);
// //     // vec3 viewport_upper_left = vec3(-viewport_width/2, viewport_height/2, -focal_length);

// //     // vec3 camera_center = vec3(0, 0, 0);
// //     // vec3 pixel_delta_u = viewport_u / (image_width - 1);
// //     // vec3 pixel_delta_v = -viewport_v / (image_width - 1);
// //     // vec3 pixel00_loc = viewport_upper_left + (pixel_delta_u / 2) + (pixel_delta_v / 2);

// //     // Allocate host memory
// //     size_t fb_size = 3 * image_width * image_height * sizeof(double);
// //     double *h_output = new double[3 * image_width * image_height];

// //     // Allocate device memory
// //     double *d_output;
// //     cudaMalloc((void**)&d_output, fb_size);


// //     // Render
// //     std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
// //     // Launch kernel
// //     int tx = 32;
// //     int ty = 32;
// //     dim3 blocks(image_width / tx + 1, image_height / ty + 1);
// //     dim3 threadsPerBlock(tx, ty);
// //     renderKernel<<<blocks, threadsPerBlock>>>(image_width, image_height, d_output);
// //     //Sync device
// //     cudaDeviceSynchronize();
// //     // Copy results from device to host
// //     cudaMemcpy(h_output, d_output, image_width * image_height * sizeof(vec3), cudaMemcpyDeviceToHost);

// //     // Print the result
// //     for (int j = 0; j < image_height; ++j) {
// //         std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
// //         for (int i = 0; i < image_width; ++i) {
// //             int pixel_index = j * 3 * image_width + 3 * i;
// //             float r = h_output[pixel_index + 0];
// //             float g = h_output[pixel_index + 1];
// //             float b = h_output[pixel_index + 2];
// //             int ir = int(255.99*r);
// //             int ig = int(255.99*g);
// //             int ib = int(255.99*b);
// //             std::cout << ir << " " << ig << " " << ib << "\n";
// //         }
// //     }

// //     std::clog << "\rDone.                 \n";
// //     // Free device memory
// //     cudaFree(d_output);

// //     // Free host memory
// //     delete[] h_output;
// //     return 0;
// // }

// int main() {

//     // Image
//     double aspect_ratio = 16.0/9.0;
//     int image_width = 256;
//     int image_height = static_cast<int>(image_width / aspect_ratio);
//     image_height = (image_height < 1) ? 1 : image_height;

//     // Allocate host memory
//     size_t fb_size = 3 * image_width * image_height * sizeof(double);
//     double *h_output = new double[3 * image_width * image_height];
    
//     // Allocate device memory
//     double *d_output;
//     checkCudaErrors(cudaMalloc((void**)&d_output, fb_size));

//     // Render

//     std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

//     // Launch kernel
//     int tx = 32;
//     int ty = 32;
//     dim3 blocks(image_width / tx + 1, image_height / ty + 1);
//     dim3 threadsPerBlock(tx, ty);
//     renderKernel<<<blocks, threadsPerBlock>>>(image_width, image_height, d_output);
//     //Sync device
//     checkCudaErrors(cudaGetLastError());
//     checkCudaErrors(cudaDeviceSynchronize());
//     // Copy results from device to host
//     checkCudaErrors(cudaMemcpy(h_output, d_output, image_width * image_height * sizeof(vec3), cudaMemcpyDeviceToHost));

//     for (int j = 0; j < image_height; ++j) {
//         std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
//         for (int i = 0; i < image_width; ++i) {
//             int pixel_index = j * 3 * image_width + 3 * i;
//             float r = h_output[pixel_index + 0];
//             float g = h_output[pixel_index + 1];
//             float b = h_output[pixel_index + 2];

//             int ir = static_cast<int>(255.999 * r);
//             int ig = static_cast<int>(255.999 * g);
//             int ib = static_cast<int>(255.999 * b);

//             std::cout << ir << ' ' << ig << ' ' << ib << '\n';
//         }
//     }
//     std::clog << "\rDone.                 \n";
// }

__global__ void render(float *fb, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x*3 + i*3;
    fb[pixel_index + 0] = float(i) / max_x;
    fb[pixel_index + 1] = float(j) / max_y;
    fb[pixel_index + 2] = 0.2;
}

int main(){
    int nx = 200;
    int ny = 100;
    int num_pixels = nx*ny;
    size_t fb_size = 3*num_pixels*sizeof(float);

    // allocate FB
    float *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
    int tx = 8;
    int ty = 8;

    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render<<<blocks, threads>>>(fb, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*3*nx + i*3;
            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    checkCudaErrors(cudaFree(fb));

    return 0;
}

