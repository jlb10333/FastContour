#include <cmath>
#include <opencv2/core/mat.hpp>
#include <optional>
#include <algorithm>
#include <iterator>
#include <iostream>
#include "../headers/utils.h"

float distance_2d(const cv::Vec2f& point1, const cv::Vec2f& point2) {
    return sqrt(pow(point1[0]-point2[0], 2) + pow(point1[1]-point2[1], 2));
}

cv::Vec2f midpoint_2d(const cv::Vec2f& point1, const cv::Vec2f& point2) {
    return cv::Vec2f(abs(point1[0]+point2[0])/2, abs(point1[1]+point2[1])/2);
}

// line intercept math by Paul Bourke (originally written in JavaScript) http://paulbourke.net/geometry/pointlineplane/
// Determine the intersection point of two line segments
// Return nullopt if the lines don't intersect
std::optional<cv::Vec2f> intersect(const cv::Vec2f& pt1, const cv::Vec2f& pt2, const cv::Vec2f& pt3, const cv::Vec2f& pt4) {
    // Check if none of the lines are length 0
    if( (pt1[0] == pt2[0] && pt1[1] == pt2[1]) || (pt3[0] == pt4[0] && pt3[1] == pt4[1])) {
        return std::nullopt;
    }

    float denominator = (pt4[1] - pt3[1]) * (pt2[0] - pt1[0]) - (pt4[0] - pt3[0]) * (pt2[1] - pt1[1]);

    // Lines are parallel
    if(float_equal(denominator, 0)) {
        return std::nullopt;
    }

    float ua = ((pt4[0] - pt3[0]) * (pt1[1] - pt3[1]) - (pt4[1] - pt3[1]) * (pt1[0] - pt3[0])) / denominator;
 
    float x = pt1[0] + ua * (pt2[0] - pt1[0]);
    float y = pt1[1] + ua * (pt2[1] - pt1[1]);

    return cv::Vec2f(x, y);
}

bool float_equal(const float left, const float right) {
    return abs(left - right) <= FLT_EPSILON;
}