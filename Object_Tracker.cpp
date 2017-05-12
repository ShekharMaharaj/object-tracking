//
//  Assignment2.cpp
//
//
//  Created by Shekhar Maharaj on 01/04/15.
//
//

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//variables to get object trajectory
int const max_images_to_track = 3;
int skipFrames = 20;
int frameCount = skipFrames-1;
int minShift = 5;
vector<vector<Point2f> > obj_points(max_images_to_track, vector<Point2f>(1));
vector<int> pointNums(max_images_to_track, 0);
vector<int> frameCounts(max_images_to_track, skipFrames-1);


//draw movement of object throughtout video
void drawyTrajectory(Mat& img, Point2f centre, int index) {
    frameCounts[index]++;
    
    //calculate shift from previous position
    double difX = abs(centre.x - obj_points[index][pointNums[index]-1].x);
    double difY = abs(centre.y - obj_points[index][pointNums[index]-1].y);
    
    //only add new point if moved significantly
    if(difX>minShift || difY>minShift) {
        //update frames every skipFrames (20)
        if(frameCounts[index]%skipFrames == 0) {
            //add new point
            obj_points[index][pointNums[index]] = centre;
            pointNums[index]++;
            //resize vector
            obj_points[index].resize(pointNums[index]+1);
        }
    }
    //set colour of object trajectory graph
    int red, green, blue;
    if(index == 0) {
        red = 255;green = 140;blue = 0;
    } else if(index == 1) {
        red = 30;green = 144;blue = 255;
    } else if(index == 2) {
        red = 154;green = 205;blue = 50;
    }
    
    //draw lines mapping trajectory
    for(int i=0; i < obj_points[index].size()-1; i++) {
        if(obj_points[index][i+1].x !=0 && obj_points[index][i+1].y!=0) {
            line(img, obj_points[index][i], obj_points[index][i+1], Scalar(blue, green, red), 2);
        }
    }
}

//track and outline object
Mat match(Mat& img1, Mat& img2) {
    
    //object to be tracked and video frame to find object in
    Mat img1Gray;
    Mat img2Gray;
    
    //convert images to grayscale for easier feature detection
    cvtColor(img1, img1Gray, CV_BGR2GRAY);
    cvtColor(img2, img2Gray, CV_BGR2GRAY);
    int minHessian = 400;
    
    //OpenCV feature detection
    SurfFeatureDetector detector(minHessian);
    
    //find key points of features
    vector<KeyPoint> keypoints1, keypoints2;
    detector.detect(img1Gray, keypoints1);
    detector.detect(img2Gray, keypoints2);
    
    //find descriptors
    SurfDescriptorExtractor extractor;
    Mat descriptors1, descriptors2;
    extractor.compute(img1Gray, keypoints1, descriptors1);
    extractor.compute(img2Gray, keypoints2, descriptors2);
    
    //OpenCV matching
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    
    //determine spread of descriptors
    double maxDist = 0;
    double minDist = 100;
    
    for(int i = 0; i < descriptors1.rows; i++) {
        double dist = matches[i].distance;
        if(dist < minDist) {
            minDist = dist;
        }
        if(dist > maxDist) {
            maxDist = dist;
        }
    }
    
    //finds good matches within reasonable area
    vector<DMatch> goodMatches;
    
    for(int i = 0; i < descriptors1.rows; i++) {
        if(matches[i].distance < 3*minDist) {
            goodMatches.push_back(matches[i]);
        }
    }
    
    //collect matches for drawing of outline
    Mat imgMatches = img2;
    
    vector<Point2f> obj;
    vector<Point2f> scene;
    
    for(int i = 0; i < goodMatches.size(); i++) {
        obj.push_back(keypoints1[goodMatches[i].queryIdx ].pt);
        scene.push_back(keypoints2[goodMatches[i].trainIdx ].pt);
    }
    
    //draw an outline around tracked object
    Mat H = findHomography(obj, scene, CV_RANSAC);
    
    vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint(img1.cols, 0);
    obj_corners[2] = cvPoint(img1.cols, img1.rows);
    obj_corners[3] = cvPoint(0, img1.rows);
    vector<Point2f> scene_corners(4);
    
    perspectiveTransform(obj_corners, scene_corners, H);
    
    line(imgMatches, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
    line(imgMatches, scene_corners[1], scene_corners[2], Scalar( 0, 255, 0), 4);
    line(imgMatches, scene_corners[2] , scene_corners[3], Scalar( 0, 255, 0), 4);
    line(imgMatches, scene_corners[3] , scene_corners[0], Scalar( 0, 255, 0), 4);
    
    //using all corners of drawn outline, find centre of outline box for graphing of object trajectory
    double sumX=0;
    double sumY=0;
    int size = scene_corners.size();
    for(int i=0;i<size;i++) {
        sumX+= scene_corners[i].x;
        sumY+= scene_corners[i].y;
    }
    
    //draw trajectory using centre point of matched object
    Point2f centre = Point2f(sumX/size, sumY/size);
    drawyTrajectory(imgMatches, centre, 0);
    
    return imgMatches;
}

//prepare video frames for analysis and output new video with tracking graphics
int processFrames(Mat& img, String source) {
    VideoCapture cap(source);
    if(!cap.isOpened()) {
        cout<<"Error opening file reader"<<endl;
        return -1;
    }

    VideoWriter writer;
    String name = "/Users/shekharmaharaj/Documents/ComputerVisionXcode/output.avi";
    //video format and settings
    int fcc = CV_FOURCC('m', 'p', '4', 'v');
    int fps = 12;
    Size frameSize(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    
    writer = VideoWriter(name, fcc, fps, frameSize);
    
    if(!writer.isOpened()) {
        cout<<"Error opening file writer"<<endl;
        return -1;
    }
    
    //loop through all frames of input video
    namedWindow("Frames",1);
    for(;;) {
        
        Mat mtch;
        Mat frame;
        cap >> frame;
        if(!frame.data) {
            break;
        }
        //show output and write to new video file
        mtch = match(img, frame);
        imshow("Frames", mtch);
        writer.write(mtch);
        if(waitKey(10) >= 0) {
            break;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    String source = "/Users/shekharmaharaj/Documents/ComputerVisionXcode/clip_sample.avi";
    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    processFrames(img1, source);
    return 0;
}
