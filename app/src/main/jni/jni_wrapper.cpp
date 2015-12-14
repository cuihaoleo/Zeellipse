#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "image_process.hpp"

using namespace std;
using namespace cv;

#ifndef JNIEXPORT
#define JNIEXPORT
#define JNICALL
#endif

extern "C" {
JNIEXPORT void JNICALL Java_com_example_cuihao_zeellipse_JniWrapper_CppPreprocess(JNIEnv* env, jobject thiz, jlong p1, jlong p2) {
    preprocess(*(Mat*)p1, *(Mat*)p2);
}

JNIEXPORT void JNICALL Java_com_example_cuihao_zeellipse_JniWrapper_CppGetSobel(JNIEnv* env, jobject thiz, jlong p1, jlong p2) {
    get_sobel(*(Mat*)p1, *(Mat*)p2);
}

JNIEXPORT void JNICALL Java_com_example_cuihao_zeellipse_JniWrapper_CppQuickFindCenter(JNIEnv* env, jobject thiz, jlong im, jobject ans) {
    jclass cls = env->GetObjectClass(ans);
    jfieldID fid_x = env->GetFieldID(cls, "x", "D"),
             fid_y = env->GetFieldID(cls, "y", "D");
    //jdouble x = env->GetIntField(ans, fid_x),
    //        y = env->GetIntField(ans, fid_y);
    auto pt = quick_find_center(*(Mat*)im);
    env->SetDoubleField(ans, fid_x, pt.x);
    env->SetDoubleField(ans, fid_y, pt.y);
}

JNIEXPORT void JNICALL Java_com_example_cuihao_zeellipse_JniWrapper_CppDynamicErode(JNIEnv* env, jobject thiz, jlong p1, jobject obj) {
    jclass cls = env->GetObjectClass(obj);
    jfieldID fid_x = env->GetFieldID(cls, "x", "D"),
             fid_y = env->GetFieldID(cls, "y", "D");
    jdouble x = env->GetDoubleField(obj, fid_x),
            y = env->GetDoubleField(obj, fid_y);
    Point center((double)x, (double)y);
    dynamic_erode(*(Mat*)p1, center);

}

JNIEXPORT void JNICALL Java_com_example_cuihao_zeellipse_JniWrapper_CppDynamicDilate(JNIEnv* env, jobject thiz, jlong p1, jobject obj) {
    jclass cls = env->GetObjectClass(obj);
    jfieldID fid_x = env->GetFieldID(cls, "x", "D"),
             fid_y = env->GetFieldID(cls, "y", "D");
    jdouble x = env->GetDoubleField(obj, fid_x),
            y = env->GetDoubleField(obj, fid_y);
    Point center((double)x, (double)y);
    dynamic_dilate(*(Mat*)p1, center);
}

JNIEXPORT void JNICALL Java_com_example_cuihao_zeellipse_JniWrapper_CppGetRBox(JNIEnv* env, jobject thiz, jlong im, jobject ans) {
    jclass cls = env->GetObjectClass(ans);
    jfieldID fid_cx = env->GetFieldID(cls, "cx", "D"),
             fid_cy = env->GetFieldID(cls, "cy", "D"),
             fid_sx = env->GetFieldID(cls, "sx", "D"),
             fid_sy = env->GetFieldID(cls, "sy", "D"),
             fid_angle = env->GetFieldID(cls, "angle", "D");
    auto rb = get_rbox(*(Mat*)im);
    env->SetDoubleField(ans, fid_cx, rb.center.x);
    env->SetDoubleField(ans, fid_cy, rb.center.y);
    env->SetDoubleField(ans, fid_sx, rb.size.width);
    env->SetDoubleField(ans, fid_sy, rb.size.height);
    env->SetDoubleField(ans, fid_angle, rb.angle);
}

JNIEXPORT jdoubleArray JNICALL Java_com_example_cuihao_zeellipse_JniWrapper_CppEllipticalIntegrate(JNIEnv* env, jobject thiz, jlong im, jobject obj) {
    jclass cls = env->GetObjectClass(obj);
    jfieldID fid_cx = env->GetFieldID(cls, "cx", "D"),
             fid_cy = env->GetFieldID(cls, "cy", "D"),
             fid_sx = env->GetFieldID(cls, "sx", "D"),
             fid_sy = env->GetFieldID(cls, "sy", "D"),
             fid_angle = env->GetFieldID(cls, "angle", "D");
    jdouble cx = env->GetDoubleField(obj, fid_cx),
            cy = env->GetDoubleField(obj, fid_cy),
            sx = env->GetDoubleField(obj, fid_sx),
            sy = env->GetDoubleField(obj, fid_sy),
            angle = env->GetDoubleField(obj, fid_angle);
    RotatedRect box(Point((double)cx, (double)cy), Size((double)sx, (double)sy), (double)angle);
    vector<double> avg;

    elliptical_integrate(*(Mat*)im, box, avg);

    jdoubleArray ans;
    ans = env->NewDoubleArray(avg.size());
    if (ans == NULL)
        return NULL;

    jdouble buf[avg.size()];
    for (int i=0; i<avg.size(); i++)
        buf[i] = avg[i];

    env->SetDoubleArrayRegion(ans, 0, avg.size(), buf);
    return ans; 
}
}
