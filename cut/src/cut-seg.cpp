#include "util.hpp"
#include "segmenter.hpp"
#include "ndarray.hpp"

// const auto sigmas = {10., 50., 100., 500., 1000., 2000., 5000.};
// const auto lambdas = {1., 1.5, 2., 2.5, 3.};
// const auto resolutions = {1000u};

int batch_segment(
        Segmenter* segmenter,
        unsigned dims,
        const unsigned* shape,
        std::byte* in_data,
        const unsigned* in_strides,
        std::byte* out_data,
        const unsigned* out_strides) {
    std::cout << "Segmenting using " << segmenter->name() << '\n';
    const NDArray<double> in_image{in_data, dims, shape, in_strides};
    NDArray<long> out_image{out_data, dims, shape, out_strides};

    // segment single image
    if (dims == 2) {
        std::cout << "segmenting single image ... "; std::cout.flush();
        segmenter->segment(in_image, out_image);
        std::cout << "done" << std::endl;
    // segment batch of images
    } else if (dims == 3) {
        std::cout << "segmenting batch of " << in_image.shape(0) << " images\n";
        for (unsigned i = 0; i < in_image.shape(0); ++i) {
            std::cout << "\tsegmenting image " << i+1 << '/' << in_image.shape(0) << " ... ";
            std::cout.flush();
            NDArray<long> out_slice = out_image.slice(i);
            segmenter->segment(in_image.slice(i), out_slice);
            std::cout << "done" << std::endl;
        }
    } else {
        std::cout << "cannot segment " << dims << "-dimensional batch\n";
        return 1;
    }
    return 0;
}

extern "C" int rbf_log_segment(
        double sigma,
        double lambda,
        unsigned resolution,
        unsigned dims,
        const unsigned* shape,
        std::byte* in_data,
        const unsigned* in_strides,
        std::byte* out_data,
        const unsigned* out_strides) {
    RBFLogSegmenter segmenter{sigma, lambda, resolution};
    return batch_segment(&segmenter, dims, shape, in_data, in_strides, out_data, out_strides);
}
