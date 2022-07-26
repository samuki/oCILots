#include "util.hpp"
#include "segmenter.hpp"

extern "C" int numpy(const int* const arr, int w, int h) {
    std::cout << w << " x " << h << " numpy array looks like\n";
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            std::cout << arr[i*h + j] << ' ';
        }
        std::cout << '\n';
    }
    std::cout.flush();
    return 0;
}

extern "C" int rbf_log_segment(double sigma, double lambda, unsigned resolution) {
    RBFLogSegmenter segmenter{sigma, lambda, resolution};
    std::cout << "segmenting using " << (&segmenter)->name() << std::endl;
    return 0;
}

void batch_segmentation() {
    const auto total_start = std::chrono::high_resolution_clock::now();

    const auto sigmas = {10., 50., 100., 500., 1000., 2000., 5000.};
    const auto lambdas = {1., 1.5, 2., 2.5, 3.};
    const auto resolutions = {1000u};
    std::vector<Segmenter*> segmenters;
    segmenters.reserve(sigmas.size() * lambdas.size() * resolutions.size());
    for (double sigma : sigmas) {
        for (double lambda : lambdas) {
            for (unsigned res : resolutions)
                segmenters.push_back(new RBFLogSegmenter(sigma, lambda, res));
        }
    }

    std::vector<OutImage> out_images(N_IMAGES, OutImage(N, -1));

    std::ios_base::sync_with_stdio(false);

    std::cout << "loading images ... "; std::cout.flush();
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<InImage> images = load_images(IN_FILENAME);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> seconds = end - start;
    std::cout << "done in " << seconds.count() << " seconds" << std::endl;

    unsigned si = 0;
    for (Segmenter* segmenter : segmenters) {
        std::cout << ++si << '/' << segmenters.size() << " segmenting with "
            << segmenter->name() << " ... ";
        std::cout.flush();
        start = std::chrono::high_resolution_clock::now();
        for (unsigned i = 0; i < images.size(); ++i) {
            segmenter->segment(images[i], out_images[i]);
        }
        save_images(out_images, segmentation_filename(segmenter->name()));
        end = std::chrono::high_resolution_clock::now();
        seconds = end - start;
        std::cout << "done in " << seconds.count() << " seconds ("
            << seconds.count() / images.size() << " per image)" << std::endl;
        delete segmenter;
    }


    seconds = end - total_start;
    std::cout << "Total: " << seconds.count() << " seconds ("
        << seconds.count() / segmenters.size() << " per segmenter)\n";
}
