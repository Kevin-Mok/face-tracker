// From https://picoledelimao.github.io/blog/2017/01/28/eyeball-tracking-for-mouse-control-in-opencv/.

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;

std::vector<Point> centers;
Point lastPoint;
Point mousePoint;

void detectFace(Mat &frame, CascadeClassifier &faceCascade)
{
    Mat grayscale;
    cvtColor(frame, grayscale, COLOR_BGR2GRAY); // convert image to grayscale
    equalizeHist(grayscale, grayscale); // enhance image contrast 
    std::vector<Rect> faces;
    faceCascade.detectMultiScale(grayscale, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(150, 150));
    if (faces.size() == 0) return; // none face was detected
    Mat face = grayscale(faces[0]); // crop the face
    rectangle(frame, faces[0].tl(), faces[0].br(), Scalar(255, 0, 0), 2);
    Point faceCenter = (faces[0].br() + faces[0].tl()) * 0.5;
    cout << faceCenter.x << " " << faceCenter.y << std::endl;
}

int main(int argc, char **argv)
{
    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade;
    if (!faceCascade.load("/home/kevin/coding/face-tracker/face-cascade.xml"))
    {
        std::cerr << "Could not load face detector." << std::endl;
        return -1;
    }    
    VideoCapture cap(0); // the fist webcam connected to your PC
    // VideoCapture cap(2); // the fist webcam connected to your PC
    // VideoCapture cap((int)argv[1]); // the fist webcam connected to your PC
    if (!cap.isOpened())
    {
        std::cerr << "Webcam " << argv[1] << " not detected." << std::endl;
        return -1;
    }    
    Mat frame;
    while (1)
    {
        cap >> frame; // outputs the webcam image to a Mat
        if (!frame.data) break;
        detectFace(frame, faceCascade);
        imshow("Face Detector", frame); // displays the Mat
        // if (waitKey(30) >= 0) break;  // takes 30 frames per second. if the user presses any button, it stops from showing the webcam
        waitKey(30);  // takes 30 frames per second. if the user presses any button, it stops from showing the webcam
    }
    return 0;
}
