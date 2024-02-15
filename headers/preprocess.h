#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

void default_preprocess_steps(
    const cv::Mat& image_in,
    cv::Mat& image_out
);

void apply_brightness_contrast(
    const cv::Mat& image_in,
    cv::Mat& image_out,
    int brightness,
    int contrast
);