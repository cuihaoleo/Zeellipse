# Zeellipse

四级大学物理实验课题，塞曼效应中的图像处理。

## 编译代码

编译前必须安装：

* Android SDK
* Android NDK

程序开发时使用的平台：

* Arch Linux x86_64
* Android Studio 1.5.1
* Android SDK 23
* Android NDK r10e

考虑到 Android 编译的复杂性，要在 Windows 编译，可能要修改不少文件。建议安装一下 Android Studio。

### 装入 OpenCV 库

到 [OpenCV 网站](http://opencv.org/) 下载 OpenCV for Android。

开发过程中适配的是 [3.0 版本](http://sourceforge.net/projects/opencvlibrary/files/opencv-android/3.0.0/OpenCV-3.0.0-android-sdk-1.zip/download)。后来发布了 3.1 版本，应该是兼容的。不要使用 2.X 版本，肯定不兼容。

将下载的 zip 文件解压，把以下文件放到本项目代码的指定目录：

* ``sdk/java`` 下的所有文件放入项目代码的 ``lib/opencv`` 目录
* ``sdk/native`` 下的所有文件放入项目代码的 ``app/src/main/jni/opencv-native/`` 目录

### 使用 Android Studio 或 Gradle 编译

本项目是从 Android Studio 1.5.1 创建的，可以直接导入。

如果没有 Android Studio，也可以用项目目录下的 Gradle 脚本来编译。比如要生成（未签名的）APK，执行（请自行替换 ``ANDROID_HOME`` 环境变量值为 ANDROID SDK 的安装目录）：

```bash
export ANDROID_HOME=/path/to/Android/SDK
./gradlew assembleRelease
```

## 应用操作

应用界面很简单。打开应用，点 *Load*，选择一张塞曼效应实验中拍摄的经F-P标准具后产生的干涉环，程序就开始自动处理图片。稍等数秒，图片加载完毕后就会显示出来。

应用有两个工作模式，可以点击左下角写着 *Auto* 或 *Manual* 的按钮切换。

*Auto Mode* 下，用户在图片控件中触摸点中一个要测量的圆环，应用就会把点击位置附近识别到的圆环画出来，并将圆环的大小、圆心位置信息显示在右下角的文本框中。

*Custom Mode* 下，用户需要在图片控件中自己触摸选择圆环上的一些点。应用会对这些点进行椭圆拟合，画出拟合出的椭圆，并把椭圆信息显示在右下角的文本框中。如果 *Auto Mode* 下自动识别失败了，可以用这个功能自己选点拟合。

左下角的 *Reset* 按钮，可以用来重置图片显示或清除手选的拟合点。

## 测试平台

* Huawei U9508, Android 4.2.2
* Genymotion's Google Nexus 4, Android 4.1.1 (with libhoudini)

在以上平台，加载图片后的计算过程耗时不超过10s。

## 已知问题

* 触摸操作的距离判定用的是图片中的像素而非物理距离，使用体验可能有些诡异
* 无效图片检查的逻辑还不太健全，加载与实验无关的图片可能导致应用崩溃
