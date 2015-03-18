#include <vector>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
	return -1;

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    int index = 0;
    char name[1000];

    vector<Mat> frames;

    for(;;)
    {
	Mat frame;
	cap >> frame; // get a new frame from camera
	frames.push_back(frame);
	if(index++>99) break;
    }
    for(int i=0; i<frames.size(); ++i) {
	sprintf(name, "input_seq/im%02d.png", i);
	imwrite(name, frames[i], compression_params);
    }
    return 0;
}

