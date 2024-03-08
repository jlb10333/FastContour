#include <optional>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <iostream>
#include "../headers/detector.h"
#include "../headers/utils.h"

#define PI 3.14159265358979323846264

using namespace cv;
using namespace std;

optional<Vec2f> find_moon_point(const Mat& img, float threshold) {
    int image_width = img.size().width;
    int image_height = img.size().height;

    float grid_size_ratio = 0.5;

    float stop_val = 0.05;

    Vec3b pixel_values;

    while(grid_size_ratio > stop_val) {
        for(int x = image_width * grid_size_ratio; x < image_width; x+= image_width * grid_size_ratio * 2) {
            for(int y = image_height * grid_size_ratio; y < image_height; y+= image_height * grid_size_ratio * 2) {
                pixel_values = img.at<Vec3b>(round(y), round(x));
                if((pixel_values[0] + pixel_values[1] + pixel_values[2])/3 >= threshold) {
                    return Vec2f(x,y);
                }
            }   
        }

        grid_size_ratio/=2;
    }
    return nullopt;
}

bool contains_point(vector<Vec2f> vec, Vec2f point) {
    for(int i = 0; i < vec.size(); i++) {
        if (vec[i][0] == point[0] && vec[i][1] == point[1]) {
            return true;
        }
    }
    return false;
}

vector<Vec2f> find_anchor_points(const Mat& img, const Vec2f& moon_point) {
    int image_width = img.size().width;
    int image_height = img.size().height;

    vector<Vec2f> anchor_points;

    Vec2f border_lines[4][2] = {
        {Vec2f(0, 0), Vec2f(image_width, 0)}, // y = 0
        {Vec2f(0, 0), Vec2f(0, image_height)}, // x = 0
        {Vec2f(0, image_height), Vec2f(image_width, image_height)}, // y = height
        {Vec2f(image_width, 0), Vec2f(image_width, image_height)} // y = width 
    };

    float angle;
    Vec2f line_point;
    optional<Vec2f> intersect_result;
    Vec2f anchor_point;
    for(int i = 0; i < 20; i++) {
        angle = (PI/10) * i;
        line_point = Vec2f(moon_point[0] + (cos(angle) * image_width * image_height), moon_point[1] + (sin(angle) * image_width * image_height));

        for(int j = 0; j < 4; j++) {
            intersect_result = intersect(moon_point, line_point, border_lines[j][0], border_lines[j][1]);

            if(!intersect_result.has_value()) { continue; }
            anchor_point = intersect_result.value();

            if(anchor_point[0] < 0 || anchor_point[0] > image_width || anchor_point[1] < 0 || anchor_point[1] > image_height) { continue; }
            if(contains_point(anchor_points, anchor_point)) { continue; }
            anchor_points.push_back(anchor_point);
        }
    }
    
    return anchor_points;
}

Vec2f regress_to_edge(const Mat& img, const Vec2f& point1, const Vec2f& point2, const float threshold) {
    Vec2f regress_point1 = Vec2f(point1[0], point1[1]);
    Vec2f regress_point2 = Vec2f(point2[0], point2[1]);

    Vec2f midpoint;
    Vec3b midpoint_pixel_values;
    Vec3b regress_pixel_values;
    while (distance_2d(regress_point1, regress_point2) > 1) {
        midpoint = midpoint_2d(regress_point1, regress_point2);

        // std::cout << "point1: " << regress_point1[0] << " " << regress_point1[1] << " " << unsigned(img.at<uint8_t>(round(regress_point1[1]), round(regress_point1[0]))) << "\n";
        // std::cout << "point2: " << regress_point2[0] << " " << regress_point2[1] << " " << unsigned(img.at<uint8_t>(round(regress_point2[1]), round(regress_point2[0]))) << "\n";

        midpoint_pixel_values = img.at<Vec3b>(round(midpoint[1]), round(midpoint[0]));
        regress_pixel_values = img.at<Vec3b>(round(regress_point1[1]), round(regress_point1[0]));
        if (((midpoint_pixel_values[0]+midpoint_pixel_values[1]+midpoint_pixel_values[2])/3 >= threshold) == ((regress_pixel_values[0]+regress_pixel_values[1]+regress_pixel_values[2])/3 >= threshold)) { 
            // midpoint and rpoint1 are on same side
            regress_point1[0] = midpoint[0];
            regress_point1[1] = midpoint[1];
        } else { 
            // midpoint and rpoint2 are on the same side
            regress_point2[0] = midpoint[0];
            regress_point2[1] = midpoint[1];
        }
    }

    return midpoint_2d(regress_point1, regress_point2);
}

pair<float, float> calculate_error(const vector<Vec2f>& point_arr, const RotatedRect& ellipse, const Size& img_size) {
    float a = ellipse.size.width/2;
    float b = ellipse.size.height/2;

    float h = ellipse.center.x;
    float k = ellipse.center.y;

    float A = ellipse.angle * 0.0174533;

    float error;

    float maximum_positive_error = 0;
    float sum_total_error = 0;
    
    for(int i = 0; i < point_arr.size(); i++) {
        Vec2f point = point_arr[i];
        float x = point[0];
        float y = point[1];
        

        error = sqrt(
            (pow(((x-h)*cos(A)) + ((y-k)*sin(A)), 2)/(a*a)) +
            (pow(((x-h)*sin(A)) - ((y-k)*cos(A)), 2)/(b*b))
            )-1;

        //cout << "point " << i << " error: " << error << "\n";

        if (error > 0 && error > maximum_positive_error) {
            maximum_positive_error = error;
        }
        sum_total_error+=abs(error);
    }

    float avg_error = sum_total_error/point_arr.size();
    float acircular_disincentive = abs((a/b) - 1);

    return pair<float, float>(maximum_positive_error, acircular_disincentive);
}

// float calculate_error(const RotatedRect& ellipse) {
//     return abs(ellipse.size.width/ellipse.size.height - 1);
// }

optional<RotatedRect> fast_contour(const Mat& img, float threshold) {
    optional<Vec2f> moon_point_result = find_moon_point(img, threshold);

    if (!moon_point_result.has_value()) {
        //cout << "empty\n";

        return nullopt;
    }

    Vec2f moon_point = moon_point_result.value();

    vector<Vec2f> anchor_points = find_anchor_points(img, moon_point);

    vector<Vec2f> edge_points;
    for (int i = 0; i < anchor_points.size(); i++) {
        edge_points.push_back(regress_to_edge(img, moon_point, anchor_points.at(i), threshold));
    }

    //cout << "edge points: ";
    // for (Vec2f i: edge_points) {
    //     cout << i << " ";
    // }
    // cout << "\n";

    float min_error = -1;
    RotatedRect detected_ellipse;
    RotatedRect best_ellipse;
    pair<float, float> ellipse_error;
    int best_index;

    float error_stop = 0.05;

    for(int i = 0; i < 20; i++) {
        vector<Vec2f> edge_point_subset;
        for(int j = 0; j < 6; j++) {
            edge_point_subset.push_back(edge_points.at((i+j) % 20));
        }

        detected_ellipse = fitEllipse(edge_point_subset);
        ellipse_error = calculate_error(edge_points, detected_ellipse, img.size());

        //cout << "index " << i << ": error " << ellipse_error.first << " " << ellipse_error.second << "\n\n";

        if(ellipse_error.first > error_stop) {continue;}

        if(min_error == -1 || ellipse_error.second < min_error) {
            min_error = ellipse_error.second;
            best_ellipse = detected_ellipse;
            best_index = i;
        }
    }

    vector<Vec2f> edge_point_subset;
    for(int i = 0; i < 6; i++) {
        edge_point_subset.push_back(edge_points.at((best_index + i) % 20));
    }

    //cout << "best index: " << best_index << "\n";

    for(int i = 0; i < 14; i++) {
        edge_point_subset.push_back(edge_points.at((best_index + 6 + i) % 20));
        detected_ellipse = fitEllipse(edge_point_subset);
        ellipse_error = calculate_error(edge_points, detected_ellipse, img.size());

        if(ellipse_error.first > error_stop || ellipse_error.second > min_error) {
            //cout << "clockwise stopping.. overall error: " << ellipse_error.second << " min error " << min_error << "\n";
            edge_point_subset.pop_back();
            break;
        }

        //cout << "including point " << (best_index + 6 + i) % 20 << "\n";
        best_ellipse = detected_ellipse;
        min_error = ellipse_error.second;
    }

    for(int i = 0; i < 14; i++) {
        edge_point_subset.push_back(edge_points.at((best_index - i + 19) % 20));
        detected_ellipse = fitEllipse(edge_point_subset);
        ellipse_error = calculate_error(edge_points, detected_ellipse, img.size());

         if(ellipse_error.first > error_stop || ellipse_error.second > min_error) {
            //cout << "counterclockwise stopping.. overall error: " << ellipse_error.second << " min error " << min_error << "\n";
            edge_point_subset.pop_back();
            break;
        }

        //cout << "including point " << (best_index + 6 + i) % 20 << "\n";
        best_ellipse = detected_ellipse;
        min_error = ellipse_error.second;
    }

    // cout << "points: ";
    // for (Vec2f i: edge_point_subset) {
    //     cout << i << " ";
    // }
    // cout << "\n";

    return best_ellipse;
}