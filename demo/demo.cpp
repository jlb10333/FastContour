#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <optional>
#include "../headers/preprocess.h"
#include "../headers/detector.h"


int main(int argc, char *argv[]) {
    if(argc < 2) {
        std::cout << "Usage: FastContour_demo <path-to-image>\n";
        return 1;
    }

    cv::Mat test_img_raw = cv::imread(argv[1], cv::IMREAD_COLOR);
    
    cv::Mat test_img_prep;
    
    default_preprocess_steps(test_img_raw, test_img_prep);
    
    cv::namedWindow("Preprocessed Image",cv::WINDOW_AUTOSIZE);
    cv::imshow("Preprocessed Image", test_img_prep);

    // Step by step run
    std::optional<cv::Vec2f> moon_point_result = find_moon_point(test_img_prep);

    if (!moon_point_result.has_value()) {
        std::cout << "Unable to detect moon in current image.\n";
        cv::waitKey(0);
        return 1;
    }

    cv::Vec2f moon_point = moon_point_result.value();
    cv::circle(test_img_raw, cv::Point(moon_point), 4, cv::Scalar(15, 255, 255), -1);

    std::vector<cv::Vec2f> anchor_points = find_anchor_points(test_img_prep, moon_point);

    std::vector<cv::Vec2f> edge_points;
    for (int i = 0; i < anchor_points.size(); i++) {
        cv::Vec2f edge_point = regress_to_edge(test_img_prep, moon_point, anchor_points.at(i));
        //cout << "Edge point " << i << ": " << edge_point[0] << " " << edge_point[1] << "\n";
        edge_points.push_back(regress_to_edge(test_img_prep, moon_point, anchor_points.at(i)));
    }

    for (int i = 0; i < edge_points.size(); i++) {
        cv::circle(test_img_raw, cv::Point(edge_points.at(i)), 4, cv::Scalar(155, 15, 155), -1);
    }

    cv::namedWindow("Edge Points",cv::WINDOW_AUTOSIZE);
    cv::imshow("Edge Points", test_img_raw);

    // Whole run
    std::optional<cv::RotatedRect> fast_contour_result = fast_contour(test_img_prep);
    if(!fast_contour_result.has_value()) {
        std::cout << "Ellipse detection failed on current image.\n";
        cv::waitKey(0);
        return 1;
    }

    cv::RotatedRect detected_ellipse = fast_contour_result.value();
    
    cv::ellipse(test_img_raw, detected_ellipse.center, cv::Size(detected_ellipse.size.width/2, detected_ellipse.size.height/2), detected_ellipse.angle, 0, 360, cv::Scalar(255, 255, 0), 3);

    cv::namedWindow("Ellipse",cv::WINDOW_AUTOSIZE);
    cv::imshow("Ellipse", test_img_raw);
    cv::waitKey(0);

    return 0;
}