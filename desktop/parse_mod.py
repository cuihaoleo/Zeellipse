#!/usr/bin/env python2

import sys
import random
import colorsys
from collections import Counter

import numpy as np
import numpy.linalg as linalg
import scipy.signal as signal
import scipy.spatial.distance as distance

import cv2
import matplotlib.pyplot as plt

if sys.version_info.major > 2:
    xrange = range

DEBUG = True


def filter_hue_val(hsv, min_val=20):
    h, s, v = cv2.split(hsv)
    h[v < min_val] = 0
    v[v < min_val] = min_val
    v -= min_val
    hist_hue, bins_hue = np.histogram(h.ravel(), 180, [0, 180])
    hist_hue[0] = 0
    peak = bins_hue[np.argmax(hist_hue)]
    mask = cv2.inRange(h, peak-10, peak+10)
    v = cv2.bitwise_and(v, mask)
    return v


def auto_resize(*imgs, **kwargs):
    width = kwargs.pop("width", 1000)

    ret = []
    for img in imgs:
        ly, lx = img.shape[:2]
        ratio = float(width) / min(lx, ly)
        ret.append(cv2.resize(img, (int(lx*ratio), int(ly*ratio))))
    return ret


def auto_crop(gray, *args, **kwargs):
    inner_crop = kwargs.pop("inner_crop", 0.08)

    buf = gray.copy()
    imgs = [gray] + list(args)

    cont, _ = cv2.findContours(buf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center, size, angle = cv2.minAreaRect(np.concatenate([cc for cc in cont]))
    m = cv2.getRotationMatrix2D(center, angle, 1)

    imgs = map(lambda i: cv2.warpAffine(i, m, (i.shape[1], i.shape[0])), imgs)
    half_ly, half_lx = np.array(size) * ((1-inner_crop) / 2.0)

    center = tuple(int(i) for i in center)
    slice_a = slice(center[1]-half_lx, center[1]+half_lx)
    slice_b = slice(center[0]-half_ly, center[0]+half_ly)
    return [i[slice_a, slice_b] for i in imgs]


def mark_connected_components(binary, min_px=100):
    ly, lx = binary.shape
    mask = np.zeros((ly+2, lx+2), np.uint8)
    flag_map = np.zeros(binary.shape, np.uint16)
    flag_count = [0]

    ill_flag = np.uint16(-1)

    flag = 0
    for y, x in np.dstack(np.nonzero(binary))[0]:
        if flag_map[y, x] == 0:
            fl = 8 | cv2.FLOODFILL_MASK_ONLY | 255 << 8
            px_count, rect = cv2.floodFill(binary, mask, (x, y), 128, flags=fl)
            if px_count > min_px:
                flag += 1
                flag_map[(flag_map == 0) & (mask[1:-1, 1:-1] != 0)] = flag
                flag_count.append(px_count)
            else:
                flag_map[(flag_map == 0) & (mask[1:-1, 1:-1] != 0)] = ill_flag

    flag_map[flag_map == ill_flag] = 0
    flag_count[0] = lx * ly - np.count_nonzero(flag_map)
    return flag_map, flag_count


def detect_ellipses(binary, flag_map, flag_count,
                    min_points=800,
                    try_merge=8,
                    max_ecc=0.4,
                    max_ra_diag_ratio=0.8):

    ly, lx = binary.shape
    max_radius = max_ra_diag_ratio * np.hypot(lx, ly) / 2.0

    def overlap_detect(ellipse_box, thickness=2):
        white = np.zeros([ly, lx], dtype=np.uint8)
        cv2.ellipse(white, ellipse_box, 255, thickness)

        overlap_flags = np.unique(flag_map[(white > 0) & (flag_map > 0)])
        overlap_flags.sort()
        overlap_count = binary[(white > 0) & (binary > 0)].shape[0]

        return overlap_count, overlap_flags

    def detect(flag):
        points = all_points[flag].copy()
        if len(points) < min_points or i == 0:
            return None

        merged = set([flag])
        for _ in range(try_merge):
            box = cv2.fitEllipse(points)
            center, size, angel = box
            ecc = np.sqrt(1 - np.square(size[0] / size[1]))
            if ecc > max_ecc:
                return None

            white = np.zeros([ly, lx], dtype=np.uint8)
            cv2.ellipse(white, box, 255, 1, cv2.CV_AA)

            count, oflags = overlap_detect(box)
            min_flag = 0
            dist = distance.cdist([center], points)[0]
            min_d = 1.2 * np.abs(dist - (size[0] + size[1])/4).max()

            for flag in oflags:
                if flag not in merged and len(all_points[flag]) <= len(points):
                    dist = distance.cdist([center], all_points[flag])[0]
                    d = np.abs(dist - (size[0] + size[1])/4).max()
                    if d < min_d:
                        min_d, min_flag = d, flag

            if min_flag == 0:
                break

            merged.add(min_flag)
            points = np.concatenate((points, all_points[min_flag]))

        _, size, _ = box
        if (size[0] + size[1]) / 4.0 >= max_radius:
            return None

        if DEBUG:
            white = np.zeros([ly, lx, 3], dtype=np.uint8)
            for flag in merged:
                white[flag_map == flag] = (0, 255, 0)
            white[flag_map == i] = (255, 255, 255)
            cv2.ellipse(white, box, (255, 0, 0), 2, cv2.CV_AA)
            cv2.imwrite("debug/contours/%d.jpg" % i, white)

        return box

    all_points = []
    for i, count in enumerate(flag_count):
        y_arr, x_arr = np.where(flag_map == i)
        all_points.append(np.array(zip(x_arr, y_arr)))

    ellipse_list = []
    for i, count in enumerate(flag_count):
        if i != 0:
            box = detect(i)
            if box:
                ellipse_list.append((count, i, box))

    return ellipse_list


def average_ellipse(el_list, center_within_ra=4):
    R = center_within_ra
    m_range = lambda c, r: xrange(int(c-R), int(c+R) + 1)

    cnt = Counter()
    for count, _, box in el_list:
        (center_x, center_y), _, _ = box
        for x in m_range(center_x, R):
            dx = x - center_x
            max_dy = np.sqrt(abs(R*R - dx*dx))
            for y in m_range(center_y, max_dy):
                cnt[(x, y)] += count

    _, max_counter = cnt.most_common(1).pop()
    select_points = [p for p, c in cnt.most_common() if c == max_counter]
    x_total = y_total = count_total = 0
    use_set = set()

    for _, flag, box in el_list:
        (center_x, center_y), size, _ = box
        for x, y in select_points:
            if np.hypot(center_x - x, center_y - y) <= R:
                use_set.add(flag)
                count_total += count
                x_total += center_x * count
                y_total += center_y * count
                break

    center = x_total / count_total, y_total / count_total

    r_table = []
    theta_list = np.radians(np.arange(0, 180, 10))
    for count, flag, box in el_list:
        if flag in use_set:
            print("Select: %d, %d, %s" % (count, flag, box))

            theta0 = np.radians(box[-1])
            b, a = (i/2.0 for i in box[1])
            r_table.append([])
            for theta in theta_list:
                dtheta = theta - theta0
                r = a*b / np.hypot(a*np.cos(dtheta), b*np.sin(dtheta))
                r_table[-1].append(r)
        else:
            print("Discard: %d, %d, %s" % (count, flag, box))

    r_table = np.array(r_table)
    pts = []
    for i, deg in enumerate(theta_list):
        r = r_table[:, i].mean()
        x = center[0] - r * np.cos(deg)
        y = center[1] - r * np.sin(deg)
        pts.append((x, y))
    box = cv2.fitEllipse(np.array(pts, dtype=np.float32))

    return box


def ra_average(gray, center=None, ra_map=None, ang_map=None, min_r=100):

    if ra_map is None:
        cx, cy = (int(c) for c in center)
        y_map, x_map = np.indices(gray.shape[:2])
        y_map -= cy
        x_map -= cx
        ra_map = np.uint32(np.hypot(x_map, y_map))
        ang_map = np.arctan2(y_map, x_map)

    r_count = np.bincount(ra_map.ravel())
    accumulator = np.vectorize(lambda r: np.sum(gray[ra_map == r]))
    r_accu = np.fromfunction(accumulator, r_count.shape)

    seq = r_accu.astype(np.float32) / r_count
    seq[:min_r] = seq[min_r]

    return seq, ra_map, ang_map


def get_elliptical_map(box, shape):
    (cx, cy), size, theta_deg = box

    theta = np.radians(theta_deg)
    b, a = np.array(size) / 2.0

    y_map, x_map = np.indices(shape).astype(np.float32)
    x_map -= cx
    y_map -= cy

    xp_map = x_map * np.cos(theta) + y_map * np.sin(theta)
    yp_map = (x_map * -np.sin(theta) + y_map * np.cos(theta)) * (b/a)
    ra_map = np.uint32(np.hypot(xp_map, yp_map))

    return ra_map


def quick_find_center(gray, it=1):
    ly, lx = gray.shape[:2]

    xi = np.arange(lx)
    yi = np.reshape(np.arange(ly), (-1, 1))

    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxdy = dx * dy
    dx2_yi = dx2 * yi
    dy2_xi = dy2 * xi
    dxdy_xi = dxdy * xi
    dxdy_yi = dxdy * yi

    ans = []
    for _ in xrange(it):
        left = np.array([
            [np.sum(dy2), -np.sum(dxdy)],
            [-np.sum(dxdy), np.sum(dx2)]
        ])
        right = np.array([
            np.sum(dy2_xi) - np.sum(dxdy_yi),
            np.sum(dx2_yi) - np.sum(dxdy_xi)
        ])

        x, y = linalg.solve(left, right)
        ans.append((x, y))
        if _ == it-1:
            break

        for yy in xrange(ly):
            for xx in xrange(lx):
                r = np.array([yy-y, xx-x], dtype=np.float64)
                d = np.array([dy[yy, xx], -dx[yy, xx]], dtype=np.float64)
                lr = linalg.norm(r)
                ld = linalg.norm(d)
                ang = np.dot(r, d) / (lr * ld)
                if ang > 0.2 or lr < 100:
                    dx2[yy, xx] = dy2[yy, xx] = dxdy[yy, xx] = \
                        dx2_yi[yy, xx] = dy2_xi[yy, xx] = \
                        dxdy_xi[yy, xx] = dxdy_yi[yy, xx] = 0.0

    return x, y


def get_sobel(gray):
    dx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, 3, scale=1)
    dy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, 3, scale=1)
    dh = np.hypot(dx, dy, dtype=np.float32)
    dd = cv2.convertScaleAbs(dh)
    dd = cv2.GaussianBlur(dd, (7, 7), 3)
    _, dd = cv2.threshold(dd, 0, 255, cv2.cv.CV_THRESH_OTSU)
    return dd


def generate_colormap(n):
    step = 1.0 / (n+1)
    for i in xrange(n):
        h = step * i
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 0.6)
        yield (int(r*255), int(g*255), int(b*255))


def continuous_elements(arr):
    i = j = 0
    while i < len(arr):
        try:
            while arr[i] == arr[j]:
                j += 1
        except IndexError:
            pass
        finally:
            yield ((i, j), arr[i])
            i = j


def erase_abnormal_conn(gray, center, ra_map, ang_map, threshold=64,
                        delta_deg=20, add_width=0.1, rmin_order=4):
    delta = np.radians(delta_deg)
    add_width = add_width * delta

    for ang in np.arange(-np.pi, np.pi, delta):
        ang_min = ang - add_width
        ang_max = ang + delta + add_width

        copy = gray.copy()
        copy[(ang_map >= ang_max) | (ang_map < ang_min)] = 0

        seq, _, _ = ra_average(copy, center, ra_map, ang_map)
        relmin = signal.argrelmin(seq, order=rmin_order)[0]

        for i in relmin:
            if seq[i] < threshold:
                gray[(ang_map <= ang_max) & (ang_map >= ang_min) &
                     (ra_map <= i+1) & (ra_map >= i-1)] = 0


def detect_turning_points(seq, threshold=-0.2,
                          min_val_avg_ratio=0.8, min_width=4):
    min_val = min_val_avg_ratio * np.mean(seq)

    grad = np.gradient(seq)
    grad2 = np.gradient(grad)

    filt = grad2 < threshold
    filt[seq < min_val] = False

    for (lo, hi), value in continuous_elements(filt):
        if not value and (hi - lo) <= min_width:
            filt[lo:hi] = True

    ranges = list(continuous_elements(filt))
    for (lo, hi), value in ranges:
        if value:
            yield int((lo + hi) / 2)


def main():
    filename = sys.argv[1]

    print("# Reading image from '%s'" % filename)
    bgr = cv2.imread(sys.argv[1])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = filter_hue_val(hsv)

    print("# Auto crop and resize...")
    cropped, cropped_bgr = auto_crop(gray, bgr)
    resized, resized_bgr = auto_resize(cropped, cropped_bgr)
    ly, lx = resized.shape[:2]

    if DEBUG:
        cv2.imwrite("debug/10_resized.jpg", resized)

    print("# Denoise...")
    denoised = cv2.fastNlMeansDenoising(resized)

    if DEBUG:
        cv2.imwrite("debug/20_denoised.jpg", denoised)

    print("# Guess center...")
    edge = cv2.Canny(denoised, 35, 45)
    edge = cv2.GaussianBlur(edge, (7, 7), 3)
    cx, cy = quick_find_center(edge)
    cv2.circle(edge, (int(cx), int(cy)), 2, 255, -1)
    print("GUESS: %d, %d" % (cx, cy))

    if DEBUG:
        cv2.imwrite("debug/30_edge.jpg", edge)

    print("# Dilate image...")
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7), (3, 3))
    dilated = cv2.dilate(denoised, element)
    eroded = cv2.erode(denoised, element)

    if DEBUG:
        cv2.imwrite("debug/40_dilated.jpg", dilated)
        cv2.imwrite("debug/45_eroded.jpg", eroded)

    print("# Filter edges...")
    sobel = get_sobel(eroded)

    if DEBUG:
        cv2.imwrite("debug/50_sobel.jpg", sobel)

    seq, ra_map, ang_map = ra_average(sobel, (cx, cy))

    print("# Erase abnormal connections...")
    erase_abnormal_conn(sobel, (cx, cy), ra_map, ang_map)

    if DEBUG:
        cv2.imwrite("debug/55_erased.jpg", sobel)

    print("# Find connected components...")
    flag_map, flag_count = mark_connected_components(sobel)
    print("Found %d connected component(s)!" % len(flag_count))

    print("# Detect ellipses...")
    el_list = detect_ellipses(sobel, flag_map, flag_count)

    print("# Calculate average ellipse...")
    center, size, theta_deg = box = average_ellipse(el_list)
    print("Center: %s, a=%.2f, b=%.2f, theta_deg = %.2f"
          % (center, size[1]/2.0, size[0]/2.0, theta_deg))

    print("# Integrate brightness on each (corrected) radius...")
    ra_map = get_elliptical_map(box, (ly, lx))
    seq, _, _ = ra_average(denoised, ra_map=ra_map)

    ra_list = list(detect_turning_points(seq))
    colormap = list(generate_colormap(len(ra_list)))
    random.shuffle(colormap)

    vis = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for ra in ra_list:
        color = colormap.pop()
        vis[(ra_map >= ra-1) & (ra_map <= ra+1)] = color
        yl, xl = np.where(ra_map == ra)
        index = random.randint(0, len(xl)-1)
        x, y = xl[index], yl[index]
        cv2.putText(vis, '%d' % ra, (x, y), font, 0.8, color, 2)

    plt.imshow(vis)
    plt.show()


if __name__ == "__main__":
    seq = main()
