#define SHOW_GUI
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include <vector>

using namespace std;
using namespace cv;

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"

#include "lo/lo.h"
#include "rokoko-common.c"

/** Function Headers */
void detectAndDisplay( cv::Mat frame, rokoko_face* );
void findFacialMarkers( cv::Mat frame, rokoko_face* );
void findFacialMarkersOld( cv::Mat frame );
void dispatch_osc(rokoko_face*);

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
cv::String face_cascade_name = "../../res/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;

#ifdef SHOW_GUI
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
std::string markers_window_name = "Capture - Facial Markers";
std::string markers_window_name_alt = "Capture - Facial Markers, alternative";
#endif

cv::RNG rng(12345);
cv::Mat debugImage;
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);

int iLowH = 47;
int iHighH = 77;
int iLowS = 82;
int iHighS = 255;
int iLowV = 64;
int iHighV = 255;
int contourAreaMin = 100;
bool rotateCam = true;
lo_address recipient;

std::string osc_address;
std::vector<cv::Rect> faces;

/**
 * @function main
 */
int main( int argc, const char** argv ) {
  CvCapture* capture;
  cv::Mat frame;

  if (argc != 4) {
    printf("Not enough arguments given.\n");
    printf("Usage: %s <camera_file> <recipient ip> <osc address>\n", argv[0]);
    return(1);
  }
  
  printf("ROKOKO Face Streamer\nJens Christian Hillerup <jc@bitblueprint.com>\nStreaming from: %s\nStreaming to:   %s\nOSC address:    %s\n", argv[1], argv[2], argv[3]);
  
  recipient = lo_address_new(argv[2], "14040");
  
  osc_address = argv[2];
  osc_address.insert(0, 1, '/');
  
  // Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -1; };

#ifdef SHOW_GUI
  cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(main_window_name, 400, 100);
  cv::namedWindow(face_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(face_window_name, 10, 100);
  cv::namedWindow(markers_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(markers_window_name, 800, 100);
  
  // Add trackbars for the HSV settings
  cvCreateTrackbar("LowH",  markers_window_name.c_str(), &iLowH, 179); //Hue (0 - 179)
  cvCreateTrackbar("HighH", markers_window_name.c_str(), &iHighH, 179);
  cvCreateTrackbar("LowS",  markers_window_name.c_str(), &iLowS, 255); //Saturation (0 - 255)
  cvCreateTrackbar("HighS", markers_window_name.c_str(), &iHighS, 255);
  cvCreateTrackbar("LowV",  markers_window_name.c_str(), &iLowV, 255); //Value (0 - 255)
  cvCreateTrackbar("HighV", markers_window_name.c_str(), &iHighV, 255);
  cvCreateTrackbar("ContourAreaMin", markers_window_name.c_str(), &contourAreaMin, 1000);
#endif
  
  createCornerKernels();
  ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2),
          43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);

  // Read the video stream
  capture = cvCaptureFromFile(argv[1]);
  if( capture ) {
    while( true ) {
      rokoko_face cur_face;
      frame = cvQueryFrame( capture );
      cv::flip(frame, frame, 1);

      if (rotateCam) {
        // Rotate by 90 degrees
        frame = frame.t();
      }

      // mirror it
      frame.copyTo(debugImage);

      // Apply the classifier to the frame
      // This is where the magic happens
      if( !frame.empty() ) {
        findFacialMarkers(frame, &cur_face);
        //findFacialMarkersOld(frame);
        detectAndDisplay(frame, &cur_face);
      }
      else {
        printf(" --(!) No captured frame -- Break!");
        break;
      }

      // Make an OSC packet with the face data.
      dispatch_osc(&cur_face);
#ifdef SHOW_GUI
      imshow(main_window_name, debugImage);

      int c = cv::waitKey(10);
      if( (char)c == 'c' ) { break; }
      if( (char)c == 'f' ) {
        imwrite("frame.png",frame);
      }
#endif
    }
  }

  releaseCornerKernels();

  return 0;
}

void dispatch_osc(rokoko_face* cur_face) {
  //pretty_print_face(cur_face);
  lo_blob blob = lo_blob_new(sizeof(rokoko_face), cur_face);
  lo_send(recipient, osc_address.c_str(), "b", blob);
}


void findFacialMarkers(cv::Mat frame, rokoko_face* cur_face) {
  vector<vector<cv::Point> > contours;
  vector<cv::Vec4i> hierarchy;
  int contours_idx = 0;

  cv::Mat imgHSV;
  cv::cvtColor(frame, imgHSV, cv::COLOR_RGB2HSV); //Convert the captured frame from RGB to HSV

  cv::Mat imgThresholded;
  cv::inRange(imgHSV, cv::Scalar(iLowH, iLowS, iLowV), cv::Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

#ifdef SHOW_GUI
  imshow(markers_window_name.c_str(), imgThresholded);
#endif
  // TODO: Apply erosion/dilation to imgThresholded?

  cv::findContours(imgThresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));

  for (int i = 0; i < contours.size(); i++) {
    if (contourArea(contours[i]) < contourAreaMin || contours_idx == MAX_CONTOURS) continue;

    Rect bound = boundingRect(contours[i]);
    Point center = Point( bound.x + (bound.width / 2), bound.y + (bound.height / 2));

    // We don't need to store imgThresholded.cols and .rows / 2 because <3 compilers.
    //cout << "(" << center.x - (imgThresholded.cols/2) << ", " << center.y - (imgThresholded.rows/2) << ")" << endl;

    circle(debugImage, center, 3, Scalar(0, 0, 255), -1);

    cur_face->contours[contours_idx++] = center;
  }

  cur_face->num_contours = contours_idx;
}

void findFacialMarkersOld(cv::Mat frame) {
  std::vector<cv::Mat> rgbChannels(3);
  cv::split(frame, rgbChannels);
  cv::Mat greens;

  int mingreen = 70;
  float gor = 1.1;

  cv::bitwise_and(rgbChannels[1] > mingreen, rgbChannels[1] > (rgbChannels[2] * gor), greens);
  cv::bitwise_and(greens, rgbChannels[1] > (rgbChannels[0] * gor), greens);

#ifdef SHOW_GUI
  imshow(markers_window_name_alt.c_str(), greens);
#endif
}

void findEyes(cv::Mat frame_gray, cv::Rect face, rokoko_face* cur_face) {
  cv::Mat faceROI = frame_gray(face);
  cv::Mat debugFace = faceROI;

  if (kSmoothFaceImage) {
    double sigma = kSmoothFaceFactor * face.width;
    GaussianBlur( faceROI, faceROI, cv::Size( 0, 0 ), sigma);
  }
  //-- Find eye regions and draw them
  int eye_region_width = face.width * (kEyePercentWidth/100.0);
  int eye_region_height = face.width * (kEyePercentHeight/100.0);
  int eye_region_top = face.height * (kEyePercentTop/100.0);
  cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
                         eye_region_top,eye_region_width,eye_region_height);
  cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
                          eye_region_top,eye_region_width,eye_region_height);

  //-- Find Eye Centers
  cv::Point leftPupil = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
  cv::Point rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");
  // get corner regions
  cv::Rect leftRightCornerRegion(leftEyeRegion);
  leftRightCornerRegion.width -= leftPupil.x;
  leftRightCornerRegion.x += leftPupil.x;
  leftRightCornerRegion.height /= 2;
  leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
  cv::Rect leftLeftCornerRegion(leftEyeRegion);
  leftLeftCornerRegion.width = leftPupil.x;
  leftLeftCornerRegion.height /= 2;
  leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
  cv::Rect rightLeftCornerRegion(rightEyeRegion);
  rightLeftCornerRegion.width = rightPupil.x;
  rightLeftCornerRegion.height /= 2;
  rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
  cv::Rect rightRightCornerRegion(rightEyeRegion);
  rightRightCornerRegion.width -= rightPupil.x;
  rightRightCornerRegion.x += rightPupil.x;
  rightRightCornerRegion.height /= 2;
  rightRightCornerRegion.y += rightRightCornerRegion.height / 2;

  #ifdef SHOW_GUI
  rectangle(debugFace,leftRightCornerRegion,200);
  rectangle(debugFace,leftLeftCornerRegion,200);
  rectangle(debugFace,rightLeftCornerRegion,200);
  rectangle(debugFace,rightRightCornerRegion,200);
  #endif
  
  // change eye centers to face coordinates
  rightPupil.x += rightEyeRegion.x;
  rightPupil.y += rightEyeRegion.y;
  leftPupil.x += leftEyeRegion.x;
  leftPupil.y += leftEyeRegion.y;

  #ifdef SHOW_GUI
  // draw eye centers
  circle(debugFace, rightPupil, 3, 1234);
  circle(debugFace, leftPupil, 3, 1234);
  #endif
  
  // Put the coordinates into our face structure
  cur_face->left_eye = leftPupil;
  cur_face->right_eye = rightPupil;

  //-- Find Eye Corners
  if (kEnableEyeCorner) {
    cv::Point2f leftRightCorner = findEyeCorner(faceROI(leftRightCornerRegion), true, false);
    leftRightCorner.x += leftRightCornerRegion.x;
    leftRightCorner.y += leftRightCornerRegion.y;
    cv::Point2f leftLeftCorner = findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
    leftLeftCorner.x += leftLeftCornerRegion.x;
    leftLeftCorner.y += leftLeftCornerRegion.y;
    cv::Point2f rightLeftCorner = findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
    rightLeftCorner.x += rightLeftCornerRegion.x;
    rightLeftCorner.y += rightLeftCornerRegion.y;
    cv::Point2f rightRightCorner = findEyeCorner(faceROI(rightRightCornerRegion), false, false);
    rightRightCorner.x += rightRightCornerRegion.x;
    rightRightCorner.y += rightRightCornerRegion.y;
#ifdef SHOW_GUI
    circle(faceROI, leftRightCorner, 3, 200);
    circle(faceROI, leftLeftCorner, 3, 200);
    circle(faceROI, rightLeftCorner, 3, 200);
    circle(faceROI, rightRightCorner, 3, 200);
#endif
  }

#ifdef SHOW_GUI  
  imshow(face_window_name, faceROI);
#endif
  //  cv::Rect roi( cv::Point( 0, 0 ), faceROI.size());
  //  cv::Mat destinationROI = debugImage( roi );
  //  faceROI.copyTo( destinationROI );
}


cv::Mat findSkin (cv::Mat &frame) {
  cv::Mat input;
  cv::Mat output = cv::Mat(frame.rows,frame.cols, CV_8U);

  cvtColor(frame, input, CV_BGR2YCrCb);

  for (int y = 0; y < input.rows; ++y) {
    const cv::Vec3b *Mr = input.ptr<cv::Vec3b>(y);
    //    uchar *Or = output.ptr<uchar>(y);
    cv::Vec3b *Or = frame.ptr<cv::Vec3b>(y);
    for (int x = 0; x < input.cols; ++x) {
      cv::Vec3b ycrcb = Mr[x];
      //      Or[x] = (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0) ? 255 : 0;
      if(skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0) {
        Or[x] = cv::Vec3b(0,0,0);
      }
    }
  }
  return output;
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( cv::Mat frame, rokoko_face* cur_face) {
  
  //cv::Mat frame_gray;

  std::vector<cv::Mat> rgbChannels(3);
  cv::split(frame, rgbChannels);
  cv::Mat frame_gray = rgbChannels[2];

  //cvtColor( frame, frame_gray, CV_BGR2GRAY );
  //equalizeHist( frame_gray, frame_gray );
  //cv::pow(frame_gray, CV_64F, frame_gray);
  //-- Detect faces
  if (faces.size() == 0) {
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
  }
  //  findSkin(debugImage);

  for( int i = 0; i < faces.size(); i++ ) {
#ifdef SHOW_GUI
    rectangle(debugImage, faces[i], 1234);
#endif
  }
  //-- Show what you got
  if (faces.size() > 0) {
    findEyes(frame_gray, faces[0], cur_face);
  }
}
