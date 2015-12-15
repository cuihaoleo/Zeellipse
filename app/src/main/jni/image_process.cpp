#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "image_process.hpp"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <cmath>
#include <cstdint>

using namespace cv;
using namespace std;

typedef vector< pair<int, RotatedRect> > ellipse_list;

inline double square(double n) {
    return n*n;
}

inline double radians(double deg) {
    return deg / 180.0 * M_PI;
}

inline double eccentricity(const RotatedRect &box) {
    double a = box.size.height,
           b = box.size.width;
    return sqrt(1.0 - (b*b)/(a*a));
    
}

Point quick_find_center(const Mat &gray) {
    Mat edges;
    Canny(gray, edges, 35, 45);
    GaussianBlur(edges, edges, Size(7, 7), 3);

    edges = gray;
    Mat mdx, mdy;
    Sobel(edges, mdx, CV_16S, 1, 0);
    Sobel(edges, mdy, CV_16S, 0, 1);

    long long a, b, d, p, q;
    long long &c = b;
    a = b = d = p = q = 0;
    for (int x=0; x<gray.cols; x++)
        for (int y=0; y<gray.rows; y++) {
            int32_t dx = mdx.at<int16_t>(y, x),
                    dy = mdy.at<int16_t>(y, x);
            a += dy*dy;
            b -= dx*dy;
            d += dx*dx;
            p += x*dy*dy - y*dx*dy;
            q += y*dx*dx - x*dx*dy;
        }

    Mat left = (Mat_<double>(2,2) << a,b,c,d);
    Mat right = (Mat_<double>(2,1) << p,q);
    Mat ans;
    solve(left, right, ans);

    return Point(ans);
}

void filter_hue_val(const Mat &hsv, Mat &dst, int min_val=20) {
    vector<Mat> hsvChannels(3);
    split(hsv, hsvChannels);

    Mat mask1;
    threshold(hsvChannels[2], mask1, 0, 255, CV_THRESH_OTSU);

    MatND hist;
    int histSize[] = {180};
    int channels[] = {0};
    float h_range[] = {0, 180};
    const float* ranges[] = {h_range};
    calcHist(&(hsvChannels[0]), 1, channels, mask1, hist, 1, histSize, ranges);
    hist.at<float>(0) = 0;

    int maxLoc[1];
    minMaxIdx(hist, 0, 0, 0, maxLoc);
    Mat mask2;
    inRange(hsvChannels[0], *maxLoc-10, *maxLoc+10, mask2);

    mask1 &= mask2;
    bitwise_and(hsvChannels[2], mask1, dst);
    dst -= min_val;
}

void auto_crop(Mat *imgs[], double inner_crop=0.08) {
    Mat gray = (*imgs[0]).clone();
    vector< vector<Point> > contours;
    findContours(gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    vector<Point> points;
    for(size_t i= 0; i < contours.size(); i++)
        for(size_t j= 0; j < contours[i].size(); j++)
            points.push_back(contours[i][j]);

    RotatedRect rect = minAreaRect(points);
    Mat m = getRotationMatrix2D(rect.center, rect.angle, 1);

    Size si = Size(rect.size.width * (1-inner_crop),
                   rect.size.height * (1-inner_crop));
    Point pa = Point(rect.center.x - si.width/2,
                     rect.center.y - si.height/2);
    Rect roi = Rect(pa, si);
    Size ws = Size(rect.center.x + si.width/2 + 1,
                   rect.center.y + si.height/2 + 1);
    for (Mat **p = imgs; *p; p++) {
        warpAffine(**p, **p, m, ws);
        Mat cropped = cv::Mat(**p, roi);
        cropped.copyTo(**p);
    }
}

void auto_resize(Mat *imgs[], int width = 1000) {
    Size si = (*imgs[0]).size();
    if (si.width > si.height)
        si = Size(si.width*(1.0*width/si.height), 1000);
    else
        si = Size(1000, si.height*(1.0*width/si.width));
    for (Mat **p = imgs; *p; p++)
        resize(**p, **p, si);
}

void get_sobel(const Mat &gray, Mat &dst) {
    Mat dx, dy;
    Sobel(gray, dx, CV_32F, 1, 0);
    Sobel(gray, dy, CV_32F, 0, 1);
    multiply(dx, dx, dx);
    multiply(dy, dy, dy);

    add(dx, dy, dst);
    sqrt(dst, dst);
    convertScaleAbs(dst, dst);
    threshold(dst, dst, 0, 255, CV_THRESH_OTSU);
}

void detect_ellipses(const Mat &bin,
                     ellipse_list &ellipses,
                     int min_pts = 500, int merge_overlap=8,
                     double overlap_detect_dt=M_PI/100.0,
                     double max_ecc=0.4) {
    Mat labels, stats, centroids;
    int nLabels = connectedComponentsWithStats(bin, labels, stats, centroids);
    vector<Point> *all_points = new vector<Point>[nLabels];

    for (int i=0; i<labels.cols; i++)
        for (int j = 0; j < labels.rows; j++) {
            int flag = labels.at<int32_t>(j, i);
            int area = stats.at<int32_t>(flag, CC_STAT_AREA);
            if (flag != 0 and area > min_pts)
                all_points[flag].push_back(Point(i, j));
        }

    for (int flag=1; flag<nLabels; flag++) {
        int area = stats.at<int32_t>(flag, CC_STAT_AREA);
        if (area < min_pts)
            continue;

        vector<Point> points = all_points[flag];
        cv::RotatedRect el = cv::fitEllipse(points);

        set<int> merged_flags;
        merged_flags.insert(0);
        merged_flags.insert(flag);
        for (int i=0; i<merge_overlap and eccentricity(el) < max_ecc; i++) {
            map<int, int> counts;
            double a = el.size.height / 2.0,
                   b = el.size.width / 2.0,
                   theta = radians(el.angle);

            // test overlap at some sample points on ellipse
            for (double t=-M_PI*2; t<M_PI*2; t+=overlap_detect_dt) {
                double x = el.center.x + b*cos(t)*cos(theta) - a*sin(t)*sin(theta),
                       y = el.center.y + b*cos(t)*sin(theta) + a*sin(t)*cos(theta);

                if (y<0 or y>=labels.rows or x<0 or x>=labels.cols)
                    continue;

                int oflag = labels.at<int32_t>((int)y, (int)x);
                if (merged_flags.find(oflag) == merged_flags.end())
                    counts[oflag]++;
            }

            int oflag = 0;
            counts[0] = 0;
            for(auto it=counts.begin(); it!=counts.end(); it++) {
                int key = it->first, value = it->second;
                if (counts[oflag] < value)
                    oflag = key;
            }

            if (oflag == 0)  // no overlap found
                break;

            vector<Point> &opts = all_points[oflag];
            if (opts.size() > points.size())  // don't merge larger area
                break;

            points.insert(points.end(), opts.begin(), opts.end());
            el = cv::fitEllipse(points);

            merged_flags.insert(oflag);
        }

        if (eccentricity(el) < max_ecc) {
            ellipses.push_back(make_pair(points.size(), el));
        }
    }

    delete[] all_points;
}

RotatedRect average_allipse(const ellipse_list ellipses, double center_within_ra=4, double dt=M_PI/20) {
    double &R = center_within_ra;
    map<pair<int, int>, int> cnt;

    int max_count = 0;
    for (auto pr: ellipses) {
        int count = pr.first;
        RotatedRect &el = pr.second;
        Point c = el.center;

        for (int x=c.x-R; x<=c.x+R; x++) {
            int dx = x - c.x;
            double max_dy = sqrt(fabs(R*R-dx*dx));
            for (int y=c.y-max_dy; y<=c.y+max_dy; y++) {
                pair<int, int> p = make_pair(x, y); 
                if ((cnt[p] += count) > max_count)
                    max_count = cnt[p];
            }
        }
    }

    set<int> selected;
    for(auto it=cnt.begin(); it!=cnt.end(); it++) {
        if (it->second == max_count) {
            int x = it->first.first,
                y = it->first.second;
            
            for (size_t i=0; i<ellipses.size(); i++) {
                const RotatedRect &el = ellipses[i].second;
                Point c = el.center;
                if (hypot(c.x-x, c.y-y) <= R)
                    selected.insert(i);
            }
        }
    }

    double x_accu, y_accu, accu;
    x_accu = y_accu = accu = 0.0;
    for (int i: selected) {
        const RotatedRect &el = ellipses[i].second;
        int count = ellipses[i].first;
        x_accu += el.center.x * count;
        y_accu += el.center.y * count;
        accu += count;
    }

    vector<Point> new_el;
    for (double t=-M_PI*2; t<M_PI*2; t+=dt) {
        double x_accu, y_accu;
        x_accu = y_accu = 0.0;

        for (int i: selected) {
            const RotatedRect &el = ellipses[i].second;
            int count = ellipses[i].first;

            double x0 = el.center.x, y0 = el.center.y;
            double dt = t-radians(el.angle);
            double b = el.size.width/2, a = el.size.height/2;
            double r = a*b / hypot(a*cos(dt), b*sin(dt));

            double x = x0 - r*cos(t),
                   y = y0 - r*sin(t);

            x_accu += x * count;
            y_accu += y * count;
        }

        new_el.push_back(Point(x_accu/accu, y_accu/accu));
    }

    RotatedRect box = fitEllipse(new_el);
    return box;
}

void _dynamic_erode_dilate(Mat &bin, Point center, double dt, uchar first, uchar second) {
    Mat new_gray(bin.size(), CV_8U);
    for (int x=0; x<bin.cols; x++)
        for (int y=0; y<bin.rows; y++) {
            int v = bin.at<uchar>(y, x);
            bool todo = (v == first);

            double rx = x - center.x,
                   ry = y - center.y;
            double r = hypot(rx, ry),
                   t = atan2(ry, rx);

            Point p1(center.x+r*cos(t+dt), center.y+r*sin(t+dt));
            LineIterator it(bin, p1, Point(x, y));
            for (int i=0; not todo and i<it.count; i++, ++it)
                if (*(uchar*)*it == first)
                    todo = true;

            Point p2(center.x+r*cos(t-dt), center.y+r*sin(t-dt));
            LineIterator it2(bin, p2, Point(x, y));
            for (int i=0; not todo and i<it2.count; i++, ++it2)
                if (*(uchar*)*it2 == first)
                    todo = true;

            new_gray.at<uchar>(y, x) = todo ? first : second;
        }

    new_gray.copyTo(bin);

}

void dynamic_erode(Mat &bin, Point center, double dt) {
    _dynamic_erode_dilate(bin, center, dt, 0, 255);
}

void dynamic_dilate(Mat &bin, Point center, double dt) {
    _dynamic_erode_dilate(bin, center, dt, 255, 0);
}

void elliptical_integrate(const Mat &gray, RotatedRect &el, vector<double> &avg) {
    double theta = radians(el.angle);
    double ratio = 1.0 * el.size.width / el.size.height;
    double cos_ = cos(theta), sin_ = sin(theta);
    double cx = el.center.x, cy = el.center.y;
    vector< array<uint32_t, 2> > mk;

    for (int x=0; x<gray.cols; x++)
        for (int y=0; y<gray.rows; y++) {
            double dx = x - cx, dy = y - cy;
            int r = (int)hypot(dx*cos_+dy*sin_, (-dx*sin_+dy*cos_)*ratio);

            if (r >= (int)mk.size())
                mk.resize(r+1);

            mk[r][0]++;
            mk[r][1] += gray.at<uchar>(y, x);
        }

    avg.clear();
    avg.reserve(mk.size());
    for (auto a: mk)
        avg.push_back((double)a[1]/a[0]);
}

void second_order_diff(vector<double> &seq, vector<double> &diff) {
    diff.clear();
    diff.resize(seq.size());
    for (size_t i=1; i<seq.size()-1; i++)
        diff[i] = seq[i+1] - 2*seq[i] + seq[i-1];
    if (diff.size() >= 2) {
        diff[0] = diff[1];
        diff[diff.size()-1] = diff[diff.size()-2];
    }
}

void preprocess(Mat &bgr, Mat &gray) {
    Mat hsv;
    cvtColor(bgr, hsv, CV_BGR2HSV);
    filter_hue_val(hsv, gray);
    Mat *gray_bgr[] = { &gray, &bgr, NULL };
    equalizeHist(gray, gray);
    auto_crop(gray_bgr);
    auto_resize(gray_bgr);
}

RotatedRect get_rbox(Mat &im) {
    ellipse_list ellipses;
    detect_ellipses(im, ellipses);
    return average_allipse(ellipses);
}

/*
int main(int argc, char** argv)
{
    Mat bgr;
    if (argc != 2 or !(bgr = imread(argv[1])).data) {
        cerr << "Please specify a valid image path." << endl;
        return -1;
    }

    Mat gray, blur, sobel;
    Point guess_center;
    RotatedRect box;

    preprocess(bgr, gray);
    GaussianBlur(gray, blur, Size(7, 7), 3);
    guess_center = quick_find_center(blur);
    get_sobel(blur, sobel);
    dynamic_erode(sobel, guess_center);
    dynamic_dilate(sobel, guess_center);
    box = get_rbox(sobel);

    vector<double> seq;
    elliptical_integrate(gray, box, seq);
    for (double i: seq)
        cout << i << ' ';
    cout << endl;

    //cout << "Center: " << box.center << ", ";
    //cout << "Size: " << box.size << ", ";
    //cout << "Angle: " << box.angle << endl;

    return 0;
}*/
