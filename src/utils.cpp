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

bool float_equal(const float left, const float right) {
    return abs(left - right) <= FLT_EPSILON;
}

////////////////////////////////////////////////////////////////////////////////
// 
// 2D Line Segment Intersection example
// Implementation of the theory provided by Paul Bourke
// 
// Written by Damian Coventry
// Tuesday, 9 January 2007
// 
////////////////////////////////////////////////////////////////////////////////

class Vector
{
public:
    float x_, y_;

    Vector(float f = 0.0f)
        : x_(f), y_(f) {}

    Vector(float x, float y)
        : x_(x), y_(y) {}
};

class LineSegment
{
public:
    Vector begin_;
    Vector end_;

    LineSegment(const Vector& begin, const Vector& end)
        : begin_(begin), end_(end) {}

    enum IntersectResult { PARALLEL, COINCIDENT, NOT_INTERESECTING, INTERESECTING };

    IntersectResult Intersect(const LineSegment& other_line, Vector& intersection)
    {
        float denom = ((other_line.end_.y_ - other_line.begin_.y_)*(end_.x_ - begin_.x_)) -
                      ((other_line.end_.x_ - other_line.begin_.x_)*(end_.y_ - begin_.y_));

        float nume_a = ((other_line.end_.x_ - other_line.begin_.x_)*(begin_.y_ - other_line.begin_.y_)) -
                       ((other_line.end_.y_ - other_line.begin_.y_)*(begin_.x_ - other_line.begin_.x_));

        float nume_b = ((end_.x_ - begin_.x_)*(begin_.y_ - other_line.begin_.y_)) -
                       ((end_.y_ - begin_.y_)*(begin_.x_ - other_line.begin_.x_));

        if(denom == 0.0f)
        {
            if(nume_a == 0.0f && nume_b == 0.0f)
            {
                return COINCIDENT;
            }
            return PARALLEL;
        }

        float ua = nume_a / denom;
        float ub = nume_b / denom;

        if(ua >= 0.0f && ua <= 1.0f && ub >= 0.0f && ub <= 1.0f)
        {
            // Get the intersection point.
            intersection.x_ = begin_.x_ + ua*(end_.x_ - begin_.x_);
            intersection.y_ = begin_.y_ + ua*(end_.y_ - begin_.y_);

            return INTERESECTING;
        }

        return NOT_INTERESECTING;
    }
};

std::optional<cv::Vec2f> intersect(const cv::Vec2f& p0, const cv::Vec2f& p1, const cv::Vec2f& p2, const cv::Vec2f& p3)
{
    LineSegment linesegment0(Vector(p0[0], p0[1]), Vector(p1[0], p1[1]));
    LineSegment linesegment1(Vector(p2[0], p2[1]), Vector(p3[0], p3[1]));

    Vector intersection;
    
    if (linesegment0.Intersect(linesegment1, intersection) == LineSegment::INTERESECTING) {
        return cv::Vec2f(std::round(intersection.x_), std::round(intersection.y_));
    }
    return std::nullopt;
}