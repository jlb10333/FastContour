#include <gtest/gtest.h>
#include <opencv2/core/mat.hpp>
#include "../headers/utils.h"

TEST(Distance2D, BasicAssertions) {
    EXPECT_EQ(distance_2d(cv::Vec2f(0, 0), cv::Vec2f(1.0, 0)), 1.0);
    EXPECT_EQ(distance_2d(cv::Vec2f(0, 0), cv::Vec2f(4.0, 3.0)), 5.0);
}

TEST(Midpoint2D, BasicAssertions) {
    EXPECT_EQ(midpoint_2d(cv::Vec2f(0, 0), cv::Vec2f(1.0, 2.0))[0], 0.5);
    EXPECT_EQ(midpoint_2d(cv::Vec2f(0, 0), cv::Vec2f(1.0, 2.0))[1], 1.0);
    EXPECT_EQ(midpoint_2d(cv::Vec2f(-2, -3), cv::Vec2f(2.0, 3.0))[0], 0);
    EXPECT_EQ(midpoint_2d(cv::Vec2f(-2, -3), cv::Vec2f(2.0, 3.0))[1], 0);
}

TEST(Intersect, FailCases) {
    EXPECT_EQ(intersect(cv::Vec2f(0, 0), cv::Vec2f(0, 0), cv::Vec2f(2, 3), cv::Vec2f(3, 3)).has_value(), false);
    EXPECT_EQ(intersect(cv::Vec2f(2, 3), cv::Vec2f(3, 3), cv::Vec2f(0, 0), cv::Vec2f(0, 0)).has_value(), false);
    EXPECT_EQ(intersect(cv::Vec2f(2, 3), cv::Vec2f(3, 3), cv::Vec2f(4, 4), cv::Vec2f(5, 4)).has_value(), false);
    EXPECT_EQ(intersect(cv::Vec2f(-2, -2), cv::Vec2f(2, 2), cv::Vec2f(-2, 2), cv::Vec2f(2, -2)).has_value(), true);
    EXPECT_EQ(intersect(cv::Vec2f(-2, -2), cv::Vec2f(-1.9, 1.9), cv::Vec2f(-2, 2), cv::Vec2f(-1.9, 1.9)).has_value(), true);
}

TEST(Intersect, BasicAssertions) {
    EXPECT_FLOAT_EQ(intersect(cv::Vec2f(-1, -2), cv::Vec2f(3, 2), cv::Vec2f(-2, 2), cv::Vec2f(2, -2)).value()[0], 0.5);
    EXPECT_FLOAT_EQ(intersect(cv::Vec2f(-1, -2), cv::Vec2f(3, 2), cv::Vec2f(-2, 2), cv::Vec2f(2, -2)).value()[1], -0.5);
    EXPECT_FLOAT_EQ(intersect(cv::Vec2f(-1, -2), cv::Vec2f(-0.9, -1.9), cv::Vec2f(-2, 2), cv::Vec2f(-1.9, 1.9)).value()[0], 0.5);
    EXPECT_FLOAT_EQ(intersect(cv::Vec2f(-1, -2), cv::Vec2f(-0.9, -1.9), cv::Vec2f(-2, 2), cv::Vec2f(-1.9, 1.9)).value()[1], -0.5);
}

TEST(FloatEqual, BasicAssertions) {
    float left = 0;
    left += 0.1;
    left += 0.1;
    left += 0.1;
    float right = 0.3;
    
    EXPECT_EQ(float_equal(left, right), true);
}