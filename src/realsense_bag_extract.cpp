#include <librealsense2/rs.hpp>
#include "cv-helpers.hpp"
#include <fstream>
#include <chrono>
#include <ctime>
#include <memory>


using namespace cv;
using namespace rs2;

Rect debugrect;
bool in_painting_rect = false;
bool start_rec = false;
bool is_start_time_set = false;

void onMouse(int event, int x, int y, int, void*)
{

    if (event == cv::EVENT_LBUTTONDOWN)
    {
        debugrect.width = 0;
        debugrect.height = 0;
        debugrect.x = x;
        debugrect.y = y;
        in_painting_rect = true;
        start_rec = false;
    }
    if (in_painting_rect && event == cv::EVENT_MOUSEMOVE)
    {
        debugrect.width = x - debugrect.x;
        debugrect.height = y - debugrect.y;
    }
    if (event == cv::EVENT_LBUTTONUP)
    {
        debugrect.width = x - debugrect.x;
        debugrect.height = y - debugrect.y;
        in_painting_rect = false;
        if(debugrect.width != 0 && debugrect.height != 0)
            start_rec = true;
    }
}

int main(int argc, char** argv) try
{
    std::ofstream fd;
    fd.open("result.txt");
    int frame_no = 0;

    std::chrono::duration<double> time_elapsed;
    std::chrono::duration<double> measure_time(7.0);    // 7 sec measuring time 
    std::chrono::system_clock::time_point start_time ; 
    std::chrono::system_clock::time_point end_time ; 

    // Start streaming from Intel RealSense bag
    pipeline pipe ;
    config cfg;
    device device; 
    frameset data;
    frame color_frame,depth_frame;
    cfg.enable_device_from_file("20190906_040324_blue_Realsense.bag");
    pipe.start(cfg);
    device = pipe.get_active_profile().get_device();
    rs2::playback playback = device.as<rs2::playback>();
    rs2::align align_to(RS2_STREAM_COLOR);

    const auto window_name = "Display Image";
    namedWindow(window_name, WINDOW_AUTOSIZE);
    
    while (getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {
        if (pipe.poll_for_frames(&data)) // Check if new frames are ready
        {
            data = align_to.process(data);
            color_frame = data.get_color_frame();
            depth_frame = data.get_depth_frame();

            // If we only received new depth frame, 
            // but the color did not update, continue
            static int last_frame_number = 0;
            if (color_frame.get_frame_number() == last_frame_number) continue;
            last_frame_number = color_frame.get_frame_number();

            // Convert RealSense frame to OpenCV matrix:
            auto color_mat = frame_to_mat(color_frame);
            auto depth_mat = depth_frame_to_meters(pipe, depth_frame);

            setMouseCallback(window_name, onMouse);
            
            //Box for recording the depth value
            rectangle(color_mat, debugrect, cv::Scalar(0, 0, 255));

            // Calculate mean depth inside the detection region
            // This is a very naive way to estimate objects depth
            // but it is intended to demonstrate how one might 
            // use depht data in general
            Scalar d = mean(depth_mat(debugrect));

            std::ostringstream label;
            label << "Average Distance: " << std::setprecision(5) << d[0] << " meters away";
            String conf(label.str());

            putText(color_mat, label.str(), Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

            imshow(window_name, color_mat);
            if (waitKey(1) >= 0) break;

            int rows = depth_mat(debugrect).rows;
            int cols = depth_mat(debugrect).cols;      

            if (start_rec == true) {
                if (is_start_time_set == false){
                    start_time = std::chrono::system_clock::now();
                    is_start_time_set = true;
                }
            }

            if (is_start_time_set == true)
                end_time = std::chrono::system_clock::now();
                time_elapsed = end_time - start_time;

            if (start_rec == true && is_start_time_set == true) {
                int i,j;
                for (i=0; i < rows; i++){
                    for (j=0; j < cols; j++){
                    fd << frame_no << " " << depth_mat.at<double>(Point(i,j)) << std::endl;
                    }
                }  
                frame_no++; 
            }
        }

        if (time_elapsed > measure_time)
            break;

    }
    std::cout << "Finish" << std::endl;
    pipe.stop();
    fd.close();
    return EXIT_SUCCESS;
}

catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}

catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}