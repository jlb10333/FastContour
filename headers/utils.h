#include <opencv2/core/mat.hpp>
#include <optional>

extern "C" 
{

typedef struct ImageShape
{
    int height;
    int width;
    int longer_side;
    int shorter_side;
} ImageShape;

float distance_2d(const cv::Vec2f& point1, const cv::Vec2f& point2);
cv::Vec2f midpoint_2d(const cv::Vec2f& point1, const cv::Vec2f& point2);
std::optional<cv::Vec2f> intersect(const cv::Vec2f& pt1, const cv::Vec2f& pt2, const cv::Vec2f& pt3, const cv::Vec2f& pt4);
bool float_equal(const float left, const float right);

}