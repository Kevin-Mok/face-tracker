// From https://picoledelimao.github.io/blog/2017/01/28/eyeball-tracking-for-mouse-control-in-opencv/.

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;

Vec3f getEyeball(Mat &eye, std::vector<Vec3f> &circles)
{
  std::vector<int> sums(circles.size(), 0);
  for (int y = 0; y < eye.rows; y++)
  {
      uchar *ptr = eye.ptr<uchar>(y);
      for (int x = 0; x < eye.cols; x++)
      {
          int value = static_cast<int>(*ptr);
          for (int i = 0; i < (int)circles.size(); i++)
          {
              Point center((int)std::round(circles[i][0]), (int)std::round(circles[i][1]));
              int radius = (int)std::round(circles[i][2]);
              if (std::pow(x - center.x, 2) + std::pow(y - center.y, 2) < std::pow(radius, 2))
              {
                  sums[i] += value;
              }
          }
          ++ptr;
      }
  }
  int smallestSum = 9999999;
  int smallestSumIndex = -1;
  for (int i = 0; i < (int)circles.size(); i++)
  {
      if (sums[i] < smallestSum)
      {
          smallestSum = sums[i];
          smallestSumIndex = i;
      }
  }
  return circles[smallestSumIndex];
}

Rect getLeftmostEye(std::vector<Rect> &eyes)
{
  int leftmost = 99999999;
  int leftmostIndex = -1;
  for (int i = 0; i < (int)eyes.size(); i++)
  {
      if (eyes[i].tl().x < leftmost)
      {
          leftmost = eyes[i].tl().x;
          leftmostIndex = i;
      }
  }
  return eyes[leftmostIndex];
}

std::vector<Point> centers;
Point lastPoint;
Point mousePoint;

Point stabilize(std::vector<Point> &points, int windowSize)
{
  float sumX = 0;
  float sumY = 0;
  int count = 0;
  for (int i = std::max(0, (int)(points.size() - windowSize)); i < (int)points.size(); i++)
  {
      sumX += points[i].x;
      sumY += points[i].y;
      ++count;
  }
  if (count > 0)
  {
      sumX /= count;
      sumY /= count;
  }
  return Point(sumX, sumY);
}

void detectEyes(Mat &frame, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade)
{
  Mat grayscale;
  cvtColor(frame, grayscale, COLOR_BGR2GRAY); // convert image to grayscale
  equalizeHist(grayscale, grayscale); // enhance image contrast 
  std::vector<Rect> faces;
  faceCascade.detectMultiScale(grayscale, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(150, 150));
  if (faces.size() == 0) return; // none face was detected
  Mat face = grayscale(faces[0]); // crop the face
  std::vector<Rect> eyes;
  eyeCascade.detectMultiScale(face, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30)); // same thing as above    
  rectangle(frame, faces[0].tl(), faces[0].br(), Scalar(255, 0, 0), 2);
  if (eyes.size() != 2) return; // both eyes were not detected
  for (Rect &eye : eyes)
  {
      rectangle(frame, faces[0].tl() + eye.tl(), faces[0].tl() + eye.br(), Scalar(0, 255, 0), 2);
  }
  Rect eyeRect = getLeftmostEye(eyes);
  Mat eye = face(eyeRect); // crop the leftmost eye
  equalizeHist(eye, eye);
  std::vector<Vec3f> circles;
  HoughCircles(eye, circles, HOUGH_GRADIENT, 1, eye.cols / 8, 250, 15, eye.rows / 8, eye.rows / 3);
  if (circles.size() > 0)
  {
      Vec3f eyeball = getEyeball(eye, circles);
      Point center(eyeball[0], eyeball[1]);
      centers.push_back(center);
      center = stabilize(centers, 5);
      if (centers.size() > 1)
      {
          Point diff;
          // diff.x = (center.x - lastPoint.x) * 20;
          // diff.y = (center.y - lastPoint.y) * -30;
          diff.x = (center.x - lastPoint.x) * 100;
          diff.y = (center.y - lastPoint.y) * -80;
          mousePoint += diff;
      }
      lastPoint = center;
      int radius = (int)eyeball[2];
      circle(frame, faces[0].tl() + eyeRect.tl() + center, radius, Scalar(0, 0, 255), 2);
      circle(eye, center, radius, Scalar(255, 255, 255), 2);
  }
  imshow("Eye", eye);
}

void changeMouse(Mat &frame, Point &location)
{
  if (location.x > frame.cols) location.x = frame.cols;
  if (location.x < 0) location.x = 0;
  if (location.y > frame.rows) location.y = frame.rows;
  if (location.y < 0) location.y = 0;
  system(("xdotool mousemove " + std::to_string(location.x) + " " + std::to_string(location.y)).c_str());
}

int main(int argc, char **argv)
{
  CascadeClassifier faceCascade;
  CascadeClassifier eyeCascade;
  if (!faceCascade.load("./face-cascade.xml"))
  {
      std::cerr << "Could not load face detector." << std::endl;
      return -1;
  }    
  if (!eyeCascade.load("./eye-cascade.xml"))
  {
      std::cerr << "Could not load eye detector." << std::endl;
      return -1;
  }
  VideoCapture cap(0); // the fist webcam connected to your PC
  if (!cap.isOpened())
  {
      std::cerr << "Webcam not detected." << std::endl;
      return -1;
  }    
  Mat frame;
  mousePoint = Point(800, 800);
  while (1)
  {
      cap >> frame; // outputs the webcam image to a Mat
      if (!frame.data) break;
      detectEyes(frame, faceCascade, eyeCascade);
      changeMouse(frame, mousePoint);
      imshow("Webcam", frame); // displays the Mat
      if (waitKey(30) >= 0) break;  // takes 30 frames per second. if the user presses any button, it stops from showing the webcam
  }
  return 0;
}
