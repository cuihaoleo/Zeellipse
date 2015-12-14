package com.example.cuihao.zeellipse;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;

public class JniWrapper {
    private static class PointStorage {
        public double x, y;
    }

    private static class RotatedBoxStorage {
        public double cx, cy, sx, sy, angle;
    }

    public static Point QuickFindCenter(Mat im) {
        PointStorage ans = new PointStorage();
        CppQuickFindCenter(im.getNativeObjAddr(), ans);
        return new Point(ans.x, ans.y);
    }

    public static RotatedRect GetRBox(Mat im) {
        RotatedBoxStorage ans = new RotatedBoxStorage();
        CppGetRBox(im.getNativeObjAddr(), ans);
        return new RotatedRect(new Point(ans.cx, ans.cy), new Size(ans.sx, ans.sy), ans.angle);
    }

    public static void DynamicErode(Mat im, Point center) {
        PointStorage c = new PointStorage();
        c.x = center.x;
        c.y = center.y;
        CppDynamicErode(im.getNativeObjAddr(), c);
    }

    public static void DynamicDilate(Mat im, Point center) {
        PointStorage c = new PointStorage();
        c.x = center.x;
        c.y = center.y;
        CppDynamicDilate(im.getNativeObjAddr(), c);
    }

    public static double[] EllipticalIntegrate(Mat im, RotatedRect box) {
        RotatedBoxStorage b = new RotatedBoxStorage();
        b.cx = box.center.x;
        b.cy = box.center.y;
        b.sx = box.size.width;
        b.sy = box.size.height;
        b.angle = box.angle;
        return CppEllipticalIntegrate(im.getNativeObjAddr(), b);
    }

    public static double EllipticalR(Point p, RotatedRect box) {
        double dx = p.x - box.center.x,
               dy = p.y - box.center.y,
               theta = Math.toRadians(box.angle),
               ratio = box.size.width / box.size.height;
        double cos_ = Math.cos(theta), sin_ = Math.sin(theta);
        return Math.hypot(dx*cos_+dy*sin_, (-dx*sin_+dy*cos_)*ratio);
    }

    public static native void CppPreprocess(long p1, long p2);
    public static native void CppGetSobel(long p1, long p2);
    public static native void CppQuickFindCenter(long p1, Object ans);
    public static native void CppDynamicErode(long p1, Object p2);
    public static native void CppDynamicDilate(long p1, Object p2);
    public static native void CppGetRBox(long p1, Object ans);
    public static native double[] CppEllipticalIntegrate(long p1, Object obj);
}