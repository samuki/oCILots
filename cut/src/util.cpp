#include "util.hpp"


std::vector<Image<double>> load_images(const std::string& filename) {
    std::vector<Image<double>> images(N_IMAGES);
    std::ifstream file{filename};
    for (unsigned n = 0; n < N_IMAGES; ++n) {
        images[n].reserve(N);
        for (unsigned i = 0; i < N; ++i) {
            double val; file >> val;
            images[n].push_back(val);
        }
    }
    file.close();
    return images;
}


void save_images(const std::vector<OutImage>& images, const std::string& filename) {
    std::ofstream file{filename};
    for (const OutImage& image : images) {
        for (unsigned i = 0; i < N; ++i) {
            file << image[i] << ' ';
        }
        file << '\n';
    }
    file.close();
}
