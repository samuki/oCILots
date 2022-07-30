#include "segmenter.hpp"
#include "ndarray.hpp"
#include <fstream>


template<typename InPixel, typename Capacity, typename OutPixel>
int batch_segment(
        Segmenter<InPixel, Capacity, OutPixel>* segmenter,
        unsigned dims,
        const unsigned* shape,
        std::byte* in_data,
        const unsigned* in_strides,
        std::byte* out_data,
        const unsigned* out_strides) {
    // std::cout << "Segmenting using " << segmenter->name() << '\n';
    const NDArray<InPixel> in_image{in_data, dims, shape, in_strides};
    NDArray<OutPixel> out_image{out_data, dims, shape, out_strides};

    // segment single image
    if (dims == 2) {
        segmenter->segment(in_image, out_image);
    // segment batch of images
    } else if (dims == 3) {
        for (unsigned i = 0; i < in_image.shape(0); ++i) {
            const NDArray<InPixel> in_slice = in_image.slice(i);
            NDArray<OutPixel> out_slice = out_image.slice(i);
            segmenter->segment(in_slice, out_slice);
        }
    } else {
        std::cout << "cannot segment " << dims << "-dimensional batch\n";
        return 1;
    }
    return 0;
}

extern "C" int rbf_log_segment_float(
        float sigma,
        float lambda,
        unsigned resolution,
        unsigned dims,
        const unsigned* shape,
        std::byte* in_data,
        const unsigned* in_strides,
        std::byte* out_data,
        const unsigned* out_strides) {
    RBFLogSegmenter<float, unsigned long long, unsigned> segmenter{sigma, lambda, resolution};
    return batch_segment<float, unsigned long long, unsigned>(&segmenter, dims, shape, in_data, in_strides, out_data, out_strides);
}

extern "C" int rbf_log_segment_double(
        double sigma,
        double lambda,
        unsigned resolution,
        unsigned dims,
        const unsigned* shape,
        std::byte* in_data,
        const unsigned* in_strides,
        std::byte* out_data,
        const unsigned* out_strides) {
    RBFLogSegmenter<double, unsigned long long, unsigned> segmenter{sigma, lambda, resolution};
    return batch_segment<double, unsigned long long, unsigned>(&segmenter, dims, shape, in_data, in_strides, out_data, out_strides);
}

extern "C" int dir_segment_unsigned(
        unsigned lambda_pred,
        unsigned lambda_dir,
        unsigned radius,
        double delta_theta,
        unsigned resolution,
        unsigned dims,
        const unsigned* shape,
        std::byte* in_data,
        const unsigned* in_strides,
        std::byte* out_data,
        const unsigned* out_strides) {
    DirectionSegmenter<unsigned, unsigned long long, unsigned> segmenter{
        lambda_pred, lambda_dir, radius, delta_theta, resolution};
    return batch_segment<unsigned, unsigned long long, unsigned>(
            &segmenter, dims, shape, in_data, in_strides, out_data, out_strides);
}
