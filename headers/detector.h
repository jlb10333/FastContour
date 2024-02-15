#include <optional>
#include <opencv2/core/mat.hpp>

extern "C" 
{

std::optional<cv::Vec2f> find_moon_point(const cv::Mat& img);
std::vector<cv::Vec2f> find_anchor_points(const cv::Mat& img, const cv::Vec2f& moon_point);
cv::Vec2f regress_to_edge(const cv::Mat& img, const cv::Vec2f& point1, const cv::Vec2f& point2);
std::optional<cv::RotatedRect> fast_contour(const cv::Mat& img);

}