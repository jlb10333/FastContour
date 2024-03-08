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
    
    cv::Mat test_img_prep = test_img_raw.clone();
    
    //default_preprocess_steps(test_img_raw, test_img_prep);
    
    cv::namedWindow("Preprocessed Image",cv::WINDOW_AUTOSIZE);
    cv::imshow("Preprocessed Image", test_img_prep);

    // Step by step run
    cv::namedWindow("test img prep before demo",cv::WINDOW_AUTOSIZE);
    cv::imshow("test img prep before demo", test_img_prep);

    std::optional<cv::Vec2f> moon_point_result = find_moon_point(test_img_prep, 30);

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
        cv::Vec2f edge_point = regress_to_edge(test_img_prep, moon_point, anchor_points.at(i), 30);
        //cout << "Edge point " << i << ": " << edge_point[0] << " " << edge_point[1] << "\n";
        edge_points.push_back(regress_to_edge(test_img_prep, moon_point, anchor_points.at(i), 30));
    }

    std::cout << "demo edge points: ";
    for (cv::Vec2f i: edge_points) {
        std::cout << i << " ";
    }
    std::cout << "\n";

    for (int i = 0; i < edge_points.size(); i++) {
        //cv::putText(test_img_raw, std::to_string(i), cv::Point(edge_points.at(i)), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(225,225,225));
        cv::circle(test_img_raw, cv::Point(edge_points.at(i)), 4, cv::Scalar(155, 15, 155), -1);
    }

    cv::namedWindow("Edge Points",cv::WINDOW_AUTOSIZE);
    cv::imshow("Edge Points", test_img_raw);

    for(int i = 0; i < 20; i++) {
        cv::Mat temp = test_img_raw.clone();

        std::vector<cv::Vec2f> edge_point_subset;
        for(int j = 0; j < 6; j++) {
            edge_point_subset.push_back(edge_points.at((i+j) % 20));
            cv::circle(temp, cv::Point(edge_points.at((i+j) % 20)), 4, cv::Scalar(15, 255, 255), -1);
        }

        cv::RotatedRect detected_ellipse = fitEllipse(edge_point_subset);

        float a = detected_ellipse.size.width/2;
        float b = detected_ellipse.size.height/2;

        float h = detected_ellipse.center.x;
        float k = detected_ellipse.center.y;

        float A = detected_ellipse.angle * 0.0174533;

        // std::cout << "points: ";
        // for (cv::Vec2f i: edge_point_subset) {
        //     std::cout << i << " ";
        // }
        // std::cout << "\n";
        // std::cout << "\ndemo, index: " << i << "; a: " << a << " b: " << b << " h: " << h << " k: " << k << " A: " << A << "\n";
        


        cv::ellipse(temp, detected_ellipse.center, cv::Size(detected_ellipse.size.width/2, detected_ellipse.size.height/2), detected_ellipse.angle, 0, 360, cv::Scalar(255, 255, 0), 3);


        
        cv::namedWindow("Ellipse " + std::to_string(i),cv::WINDOW_AUTOSIZE);
        cv::imshow("Ellipse" + std::to_string(i), temp);
    }

    // Whole run

    cv::namedWindow("test img prep before run",cv::WINDOW_AUTOSIZE);
    cv::imshow("test img prep before run", test_img_prep);

    

    std::optional<cv::RotatedRect> fast_contour_result = fast_contour(test_img_prep, 30);
    if(!fast_contour_result.has_value()) {
        std::cout << "Ellipse detection failed on current image.\n";
        cv::waitKey(0);
        return 1;
    }

    cv::RotatedRect detected_ellipse = fast_contour_result.value();
    
    cv::ellipse(test_img_raw, detected_ellipse.center, cv::Size(detected_ellipse.size.width/2, detected_ellipse.size.height/2), detected_ellipse.angle, 0, 360, cv::Scalar(255, 255, 0), 3);

    // std::vector<cv::Vec2f> edge_point_subset = {edge_points.at(16),edge_points.at(17),edge_points.at(18),edge_points.at(19),edge_points.at(0),edge_points.at(1),edge_points.at(2)};
    // edge_point_subset.push_back(edge_points.at(9));
    // cv::RotatedRect second_ellipse = cv::fitEllipse(edge_point_subset);
    // cv::ellipse(test_img_raw, second_ellipse, cv::Scalar(0,255,255));

    cv::namedWindow("Best Ellipse",cv::WINDOW_AUTOSIZE);
    cv::imshow("Best Ellipse", test_img_raw);
    cv::waitKey(0);

    return 0;
}