LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
#OPENCV_LIB_TYPE:=STATIC
include $(LOCAL_PATH)/opencv-native/jni/OpenCV.mk
LOCAL_MODULE    := jni_wrapper
LOCAL_SRC_FILES := image_process.cpp jni_wrapper.cpp
LOCAL_LDLIBS +=  -llog -ldl
include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencv_java3-prebuilt
LOCAL_SRC_FILES := opencv-native/libs/$(TARGET_ARCH_ABI)/libopencv_java3.so
include $(PREBUILT_SHARED_LIBRARY)
