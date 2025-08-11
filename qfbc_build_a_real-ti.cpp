#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <ARCore_CPP_API.h>

using namespace std;
using namespace cv;

class ARVRNotifier {
private:
    ArSession* session;
    ArFrame* frame;
    ArPointCloud* pointCloud;
    vector<KeyPoint> keypoints;
    Mat descriptors;
    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> extractor;

public:
    ARVRNotifier() {
        // Initialize ARCore session
        ArSession_create(session);
        ArSession_configure(session, ARCONFIG_ENABLE_CLOUD_ANCHORS);
        ArSession_setDisplayGeometry(session, 0, 0, 0, 1, 1);

        // Initialize OpenCV detector and extractor
        detector = ORB::create();
        extractor = ORB::create();
    }

    ~ARVRNotifier() {
        ArSession_destroy(session);
    }

    void processFrame(Mat frameRGB) {
        // Convert frame to grayscale
        Mat frameGray;
        cvtColor(frameRGB, frameGray, COLOR_BGR2GRAY);

        // Detect keypoints
        detector->detect(frameGray, keypoints);

        // Extract descriptors
        extractor->compute(frameGray, keypoints, descriptors);

        // Create an ARCore frame
        ArFrame_create(session, frame);

        // Set the camera image
        ArFrame_updateCameraImage(frame, frameRGB.data, frameRGB.cols, frameRGB.rows);

        // Detect anchors
        ArSession_detectAnchors(session, frame);

        // Get the point cloud
        ArFrame_getPointCloud(frame, &pointCloud);

        // Check for new anchors
        for (int i = 0; i < ArPointCloud_getPointCount(pointCloud); i++) {
            ArAnchor* anchor;
            if (ArPointCloud_getAnchor(pointCloud, i, &anchor)) {
                // Check if the anchor is new
                if (ArAnchor_getTrackingState(anchor) == AR_TRACKING_STATE_TRACKING) {
                    // Get the anchor's pose
                    float transform[16];
                    ArAnchor_getTransform(anchor, transform);

                    // Notify the user
                    notifyUser(transform);
                }
            }
        }
    }

    void notifyUser(float transform[16]) {
        // Use the pose to calculate the distance from the camera
        float distance = sqrt(transform[0] * transform[0] + transform[1] * transform[1] + transform[2] * transform[2]);

        // Print the distance
        cout << "New anchor detected at distance: " << distance << " meters" << endl;
    }
};

int main() {
    ARVRNotifier notifier;
    VideoCapture cap(0);

    if (!cap.isOpened()) {
        cerr << "Error opening camera" << endl;
        return 1;
    }

    while (true) {
        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            break;
        }

        notifier.processFrame(frame);
    }

    return 0;
}