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

到 [OpenCV 网站](http://opencv.org/) 下载 OpenCV for Android。3.0 和 3.1 版本都可以用。

将下载的 zip 文件解压，把以下文件放到本项目代码的指定目录：

* ``sdk/java`` 下的所有文件放入项目代码的 ``lib/opencv`` 目录
* ``sdk/native`` 下的所有文件放入项目代码的 ``app/src/main/jni/opencv-native/`` 目录

### 使用 Android Studio 或 Gradle 编译

本项目是从 Android Studio 1.5.1 创建的，可以直接导入。如果没有 Android Studio，也可以用项目目录下的 Gradle 脚本来编译。

如果 SDK、NDK 工具不在 ``$PATH`` 中，需要手动设置一下工具链的安装目录。方法是在项目目录下创建 ``local.properties``，内容形如：
```
ndk.dir=/path/to/android/ndk-bundle
sdk.dir=/path/to/android/sdk
```

也可以通过指定 ``ANDROID_HOME`` 和 ``ANDROID_NDK_ROOT`` 环境变量来设置工具链安装目录。

要生成（未签名的）APK，在项目目录执行：

```bash
./gradlew assembleRelease
```

然后在 ``app/build/outputs/apk/`` 目录下就会生成适用于各种 ABI 的若干 APK 文件。

## 应用操作

应用界面很简单。打开应用，点 *Load*，选择一张塞曼效应实验中拍摄的经F-P标准具后产生的干涉环，程序就开始自动处理图片。稍等数秒，图片加载完毕后就会显示出来。

应用有两个工作模式，可以点击左下角写着 *Auto* 或 *Manual* 的按钮切换。

*Auto Mode* 下，用户在图片控件中触摸点中一个要测量的圆环，应用就会把点击位置附近识别到的圆环画出来，并将圆环的大小、圆心位置信息显示在右下角的文本框中。

*Custom Mode* 下，用户需要在图片控件中自己触摸选择圆环上的一些点。应用会对这些点进行椭圆拟合，画出拟合出的椭圆，并把椭圆信息显示在右下角的文本框中。如果 *Auto Mode* 下自动识别失败了，可以用这个功能自己选点拟合。

左下角的 *Reset* 按钮，可以用来重置图片显示或清除手选的拟合点。

## 测试平台

* Huawei U9508, Android 4.2.2 (armeabi, armeabi-v7a)
* Genymotion's Google Nexus 4, Android 4.1.1 (x86, armeabi-v7a with libhoudini)
* Genymotion's Google Nexus 7, Android 5.1.0 (x86)

在以上平台，除了 armeabi，加载图片后的计算过程一般耗时不超过10s。

## 已知问题

* 触摸操作的距离判定用的是图片中的像素而非物理距离，使用体验可能有些诡异
* 无效图片检查的逻辑还不太健全，加载与实验无关的图片可能导致应用崩溃
