#include <iostream>
#include <string.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "face_detection.h"
#include "face_alignment.h"
#include "face_identification.h"

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {

    if(argc != 3)
    {
        cout << "Correct format: ./build/seetaface_example gallery_img_path probe_img_path" << endl;
        return 0;
    }

    // Initialize face detection model
    string detector_model_path = "model/seeta_fd_frontal_v1.0.bin";
    seeta::FaceDetection face_detector(detector_model_path.c_str());
    face_detector.SetMinFaceSize(100);
    face_detector.SetMaxFaceSize(500);
    face_detector.SetScoreThresh(2.f);
    face_detector.SetImagePyramidScaleFactor(0.8f);
    face_detector.SetWindowStep(4, 4);

    // Initialize face alignment model
    string aligner_model_path = "model/seeta_fa_v1.1.bin";
    seeta::FaceAlignment face_aligner(aligner_model_path.c_str());

    // Initialize face Identification model
    string identifier_model_path = "model/seeta_fr_v1.0.bin";
    seeta::FaceIdentification face_identifier(identifier_model_path.c_str());

    // Load image
    string gallery_img = argv[1];
    string probe_img = argv[2];

    Mat gallery_img_color = imread(gallery_img, 1);
    Mat gallery_img_gray;
    cvtColor(gallery_img_color, gallery_img_gray, CV_RGB2GRAY);

    Mat probe_img_color = imread(probe_img, 1);
    Mat probe_img_gray;
    cvtColor(probe_img_color, probe_img_gray, CV_RGB2GRAY);

    seeta::ImageData gallery_img_data_color(gallery_img_color.cols, gallery_img_color.rows, gallery_img_color.channels());
    gallery_img_data_color.data = gallery_img_color.data;

    seeta::ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
    gallery_img_data_gray.data = gallery_img_gray.data;

    seeta::ImageData probe_img_data_color(probe_img_color.cols, probe_img_color.rows, probe_img_color.channels());
    probe_img_data_color.data = probe_img_color.data;

    seeta::ImageData probe_img_data_gray(probe_img_gray.cols, probe_img_gray.rows, probe_img_gray.channels());
    probe_img_data_gray.data = probe_img_gray.data;

    // Detect faces
    vector<seeta::FaceInfo> gallery_faces = face_detector.Detect(gallery_img_data_gray);
    int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());

    std::vector<seeta::FaceInfo> probe_faces = face_detector.Detect(probe_img_data_gray);
    int32_t probe_face_num = static_cast<int32_t>(probe_faces.size());

    if (gallery_face_num == 0 || probe_face_num==0)
    {
        cout << "Faces are not detected." << endl;
        return 0;
    }

    for(auto item:gallery_faces)
    {
        Rect face_rect;
        face_rect.x = item.bbox.x;
        face_rect.y = item.bbox.y;
        face_rect.width = item.bbox.width;
        face_rect.height = item.bbox.height;
        rectangle(gallery_img_color, face_rect, CV_RGB(0, 0, 255));
        putText(gallery_img_color, std::to_string(item.score), face_rect.br(), 0, 1.0, CV_RGB(0, 0, 255));
    }

    for(auto item:probe_faces)
    {
        Rect face_rect;
        face_rect.x = item.bbox.x;
        face_rect.y = item.bbox.y;
        face_rect.width = item.bbox.width;
        face_rect.height = item.bbox.height;
        rectangle(probe_img_color, face_rect, CV_RGB(0, 0, 255));
    }

    // Detect 5 facial landmarks
    seeta::FacialLandmark gallery_points[5];
    face_aligner.PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);

    seeta::FacialLandmark probe_points[5];
    face_aligner.PointDetectLandmarks(probe_img_data_gray, probe_faces[0], probe_points);

    for (int i = 0; i<5; i++)
    {
        circle(gallery_img_color, Point(gallery_points[i].x, gallery_points[i].y), 2,
        CV_RGB(0, 255, 0));
        circle(probe_img_color, Point(probe_points[i].x, probe_points[i].y), 2,
        CV_RGB(0, 255, 0));
    }
    imshow("gallery_point_result.jpg", gallery_img_color);
    imshow("probe_point_result.jpg", probe_img_color);

    // Extract face identity feature
    float gallery_fea[2048];
    float probe_fea[2048];
    face_identifier.ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, gallery_fea);
    face_identifier.ExtractFeatureWithCrop(probe_img_data_color, probe_points, probe_fea);

    // Caculate similarity of two faces
    float sim = face_identifier.CalcSimilarity(gallery_fea, probe_fea);
    cout << "The similarity between " << gallery_img << " and " << probe_img << " is: " << sim << endl;

    waitKey(0);

    return 0;
}
