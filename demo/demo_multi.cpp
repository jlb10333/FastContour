#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <optional>
#include <iostream>
#include <fstream>
#include <vector>
#include <exception>
#include <chrono>
#include <numeric>
#include <filesystem>
namespace fs = std::filesystem;

#include "../headers/preprocess.h"
#include "../headers/detector.h"

double average(std::vector<double> const& v){
    if(v.empty()){
        return 0;
    }

    auto const count = static_cast<double>(v.size());
    return std::reduce(v.begin(), v.end()) / count;
}

int main(int argc, char *argv[])
{
    if(argc < 3) {
        std::cout << "Usage: FastContour_demo <path-to-image-folder> <path-to-output-folder>\n";
        return 1;
    }

    cv::Mat current_img;

    std::vector<double> measurements;
    measurements.reserve(200);
    auto loop_start = std::chrono::high_resolution_clock::now();
    fs::path folder(argv[1]);
    std::cout << "Folder Path: " << folder << "\n";
    for (auto dirEntry : fs::recursive_directory_iterator(folder))
    {
        if (!fs::is_regular_file(dirEntry))
            continue;
        try
        {   
            auto detect_start = std::chrono::high_resolution_clock::now();
            
            current_img = cv::imread(dirEntry.path().string(), cv::IMREAD_COLOR);
 
            std::optional<cv::RotatedRect> fast_contour_result = fast_contour(current_img, 30);
            if(!fast_contour_result.has_value()) {
                continue;
            }

            cv::RotatedRect detected_ellipse = fast_contour_result.value();

            auto detect_stop = std::chrono::high_resolution_clock::now();
            
            auto detect_duration = std::chrono::duration_cast<std::chrono::microseconds>(detect_stop - detect_start);
            measurements.push_back(
                static_cast<double>(detect_duration.count())
            );
            
            cv::ellipse(current_img, detected_ellipse.center, cv::Size(detected_ellipse.size.width/2, detected_ellipse.size.height/2), detected_ellipse.angle, 0, 360, cv::Scalar(255, 255, 0), 3);

            // std::vector<cv::Vec2f> edge_point_subset = {edge_points.at(16),edge_points.at(17),edge_points.at(18),edge_points.at(19),edge_points.at(0),edge_points.at(1),edge_points.at(2)};
            // edge_point_subset.push_back(edge_points.at(9));
            // cv::RotatedRect second_ellipse = cv::fitEllipse(edge_point_subset);
            // cv::ellipse(test_img_raw, second_ellipse, cv::Scalar(0,255,255));

            cv::imwrite(argv[2] + dirEntry.path().filename().string() + "_result" + dirEntry.path().extension().string(), current_img);
        }
        catch (const std::exception& error)
        {
            std::cerr << "Exception: " << error.what() << "\n";
            return -1;
        }
    }
    auto loop_stop = std::chrono::high_resolution_clock::now();
    auto loop_duration = std::chrono::duration_cast<std::chrono::microseconds>(loop_stop - loop_start);
    std::cout << "\n\n\n";
    std::cout << "loop_duration.count(): " << loop_duration.count() << std::endl;;
    std::cout << "average(measurements): " << average(measurements) << std::endl;;
    
    return 0;
}
