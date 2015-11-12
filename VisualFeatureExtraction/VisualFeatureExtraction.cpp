// VisualFeatureExtraction.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/objdetect/objdetect.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"
#include "opencv\cv.h"
#include "opencv\highgui.h"
#include <iostream>

using namespace cv;
using namespace std;

 void detectAndDisplay( Mat frame );
 void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
	                    double scale, const Scalar& color)
	{
	    for(int y = 0; y < cflowmap.rows; y += step)
	        for(int x = 0; x < cflowmap.cols; x += step)
	        {
	            const Point2f& fxy = flow.at<Point2f>(y, x);
	            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
	                 color);
	            circle(cflowmap, Point(x,y), 2, color, -1);	        }
	}

String face_cascade_name = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
 String eyes_cascade_name = "C:/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
String mouth_cascade_name="C:/opencv/sources/data/haarcascades/haarcascade_mcs_mouth.xml";
 
 CascadeClassifier face_cascade;
 CascadeClassifier eyes_cascade;
 CascadeClassifier mouth_cascade;
int main(int argc, char* argv[])
{
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -2; };
   if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   if( !mouth_cascade.load( mouth_cascade_name ) ){ printf("--(!)Error loading\n"); return -3; };
    VideoCapture cap("C:/Users/Arjun/Videos/01-01-01-01-01.avi"); // open the video camera no. 0

    if (!cap.isOpened())  // if not success, exit program
    {
        cout << "Cannot open the video cam" << endl;
        return -4;
    }
	

   double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
   double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

    cout << "Frame size : " << dWidth << " x " << dHeight << endl;

  //  namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
/*
    while (1)
    {
        Mat frame;

        bool bSuccess = cap.read(frame); // read a new frame from video

         if (!bSuccess) //if not success, break loop
        {
             cout << "Cannot read a frame from video stream" << endl;
             break;
        }
		  if( !frame.empty() )
       { detectAndDisplay( frame ); }
       else
       { printf(" --(!) No captured frame -- Break!"); break; }

       int c = waitKey(10);
       if( (char)c == 'c' ) { break; }
       // imshow("MyVideo", frame); //show the frame in "MyVideo" window

        if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
       {
            cout << "esc key is pressed by user" << endl;
            break; 
       }
    }
    return 0;
*/
	Mat prevgray, gray, flow, cflow, frame;
	    namedWindow("flow", 1);
	   bool bSuccess = cap.read(frame);
	   
	   if(bSuccess)
	   {
		   for(;;)
	    {
	        cap >> frame;
	        cvtColor(frame, gray, CV_BGR2GRAY);
	       
	        if( prevgray.data )
	        {
	            calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
	            cvtColor(prevgray, cflow, CV_GRAY2BGR);
	            drawOptFlowMap(flow, cflow, 16, .5, CV_RGB(0, 255, 0));
	            imshow("flow", cflow);
	        }
	        if(waitKey(30)>=0)
	            break;
	        std::swap(prevgray, gray);
	    }
	   }
	    return 0;
}
void detectAndDisplay( Mat frame )
{
  std::vector<Rect> faces;
  Mat frame_gray;

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
	
		
    Mat faceROI = frame_gray( faces[i] );
    std::vector<Rect> eyes;

    //-- In each face, detect eyes
   eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    for( size_t j = 0; j < eyes.size(); j++ )
     {
       Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
       int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
       circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
	 //  	Mat MoutROI =  frame_gray( faces[i] );;
	
  
     }
	 std::vector<Rect> mouth;
  mouth_cascade.detectMultiScale( faceROI, mouth, 1.1, 1, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
  for( size_t k = 0; k < mouth.size(); k++ )
     {
       Point center1(faces[i].x+mouth[k].x + mouth[k].width*0.5,faces[i].y+ mouth[k].y + mouth[k].height*0.5 );
       int radius = cvRound( (mouth[k].width + mouth[k].height)*0.25 );
       circle( frame, center1, radius, Scalar( 0, 0,222 ), 4, 8, 0 );
	   cout<<k;
     }
  }
 
  //-- Show what you got
  imshow( "MyVideo", frame );
 }
