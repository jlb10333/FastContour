#include <gtest/gtest.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "../headers/detector.h"
#include "../headers/preprocess.h"

TEST(FindMoonPoint, FailsCorrectly) {
    cv::Mat test_img = cv::Mat::zeros(cv::Size(200,200), CV_8UC1);

    EXPECT_EQ(find_moon_point(test_img).has_value(), false);
}

TEST(FindMoonPoint, SucceedsCorrectlyFirstRound) {
    cv::Mat test_img = cv::Mat::zeros(cv::Size(200,200), CV_8UC1);
    cv::circle(test_img, cv::Point(100, 100), 50, cv::Scalar(255), -1);

    cv::Vec2f result_point = find_moon_point(test_img).value();

    EXPECT_EQ(test_img.at<cv::uint8_t>(std::round(result_point[1]), std::round(result_point[0])), 255);
}

TEST(FindMoonPoint, IteratesCorrectly) {
    cv::Mat test_img = cv::Mat::zeros(cv::Size(200,200), CV_8UC1);
    cv::circle(test_img, cv::Point(135, 147), 30, cv::Scalar(255), -1);

    cv::Vec2f result_point = find_moon_point(test_img).value();

    EXPECT_EQ(test_img.at<cv::uint8_t>(std::round(result_point[1]), std::round(result_point[0])), 255);
}

TEST(FindAnchorPoints, BasicAssertions) {
    cv::Mat test_img = cv::Mat::zeros(cv::Size(200,200), CV_8UC1);

    std::vector<cv::Vec2f> anchor_points = find_anchor_points(test_img, cv::Vec2f(100, 100));
    std::cout << anchor_points[0] << "\n";
    EXPECT_EQ(anchor_points.size(), 16);
}

TEST(RegressToEdge, BasicAssertions) {
    cv::Mat test_img = cv::Mat::zeros(cv::Size(200,200), CV_8UC1);
    cv::circle(test_img, cv::Point(100, 100), 50, cv::Scalar(255), -1);

    cv::Vec2f edge_point = regress_to_edge(test_img, cv::Vec2f(0, 100), cv::Vec2f(100, 100));

    EXPECT_EQ(round(edge_point[0]), 50);
    EXPECT_EQ(round(edge_point[1]), 100);

    cv::Vec2f edge_point2 = regress_to_edge(test_img, cv::Vec2f(100, 0), cv::Vec2f(100, 100));

    EXPECT_EQ(round(edge_point2[0]), 100);
    EXPECT_EQ(round(edge_point2[1]), 50);
}

TEST(FastContour, BasicAssertions) {

}