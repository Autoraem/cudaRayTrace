#include "color.cuh"
#include "vec3.cuh"
#include "ray.cuh"
#include "camera.cuh"
#include <iostream>


int main() {
    Camera cam;
    // Render
    std::cout << "P3\n" << cam.image_width << ' ' << cam.image_height << "\n255\n";
    cam.render();

    // Print the result
    for (int j = 0; j < cam.image_height; ++j) {
        std::clog << "\rScanlines remaining: " << (cam.image_height - j) << ' ' << std::flush;
        for (int i = 0; i < cam.image_width; ++i) {
            int pixel_index = j * 3 * cam.image_width + 3 * i;
            float r = cam.h_output[pixel_index + 0];
            float g = cam.h_output[pixel_index + 1];
            float b = cam.h_output[pixel_index + 2];
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
            std::cout << ir << " " << ig << " " << ib << "\n";
            //write_color(std::cout, cam.h_output[j * cam.image_width + i]);
        }
    }

    // Free memory/deconstruct
    cam.deleteCamera();

    std::clog << "\rDone.                 \n";

    return 0;
}



