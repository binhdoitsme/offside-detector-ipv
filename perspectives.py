from __future__ import annotations
import math

from typing import Optional, Union

import cv2
import numpy as np


RhoThetaLine = tuple[float, ...]
Point = tuple[int, ...]
PointsLine = Union[list[Point], tuple[Point, ...], dict[str, tuple[float, ...]]]

debug = False


def is_in_image(p: tuple[int, int], img_h: int, img_w: int):
    return 0 <= p[0] < img_w and 0 <= p[1] < img_h


def is_leftside_point(point: Point, line: tuple[Point, ...]) -> bool:
    x, y = point
    (x1, y1), (x2, y2) = line
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1) < 0


def intersect(line1: PointsLine, line2: PointsLine):
    if isinstance(line1, (list, tuple)) and isinstance(line2, (list, tuple)):
        p1, p2, p3, p4 = line1[0], line1[1], line2[0], line2[1]
    elif isinstance(line1, (dict)) and isinstance(line2, (dict)):
        p1, p2, p3, p4 = line1["p1"], line1["p2"], line2["p1"], line2["p2"]
    else:
        raise ValueError("line1 and line2 are of different types")

    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4

    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    )
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    )
    return x, y


def are_same_line(line1: RhoThetaLine, line2: RhoThetaLine):
    rho1, theta1 = line1
    rho2, theta2 = line2
    return abs(rho1 - rho2) < 20 and abs(theta1 - theta2) < 0.4


def get_vanishing_point(
    img: cv2.Mat, sort_lines=False, sort_lines_desc=False
) -> tuple[int, int]:
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    edges = cv2.Canny(blur_gray, 25, 100, apertureSize=3)

    if debug:
        cv2.imshow("edges", edges)

    all_lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if sort_lines:
        by_theta = lambda line: line[0][1]
        all_lines = list(sorted(all_lines, key=by_theta, reverse=sort_lines_desc))

    parallel_lines = []

    first_rho: Optional[float] = None
    first_theta: Optional[float] = None

    im_debug = img.copy() if debug else img

    for line in all_lines:
        for (rho, theta) in line:
            # sin(theta) = 0 -> vertical line; cos(theta) = 0 -> horizontal line
            if math.sin(theta) == 0 or math.cos(theta) == 0:
                continue

            # 2 parallel lines are enough for estimation
            if len(parallel_lines) == 2:
                break

            # skip possibly duplicate line
            skip_this_line = (
                len(parallel_lines) == 1
                and first_theta is not None
                and first_rho is not None
                and are_same_line((rho, theta), (first_rho, first_theta))
            )
            if skip_this_line:
                continue

            # find 2 point of the given line to draw by OpenCV
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            if debug:
                cv2.line(im_debug, (x1, y1), (x2, y2), (0, 255, 255), 2)

            first_rho, first_theta = rho, theta
            parallel_lines.append({"p1": (x1, y1), "p2": (x2, y2)})

            try:
                if len(parallel_lines) == 2:
                    # calculate intersection of 2 lines and compare to threshold
                    intersect_x, intersect_y = intersect(*parallel_lines)
                    if (
                        intersect_y > 0
                        or intersect_x < -1.25 * img.shape[1]
                        or intersect_x > 1.5 * img.shape[1]
                    ):
                        parallel_lines.pop()
            except Exception as e:
                print("exception:", e)
                ## lines are parallel
                parallel_lines.pop()

    if debug:
        cv2.imshow("", im_debug)

    # retry using line sorting (asc)
    if len(parallel_lines) != 2 and not sort_lines:
        return get_vanishing_point(img, sort_lines=True)

    # retry using line sorting (desc)
    if len(parallel_lines) != 2 and sort_lines and not sort_lines_desc:
        return get_vanishing_point(img, sort_lines=True, sort_lines_desc=True)

    # all else fail then no vanishing point can be estimated
    if len(parallel_lines) != 2:
        raise ValueError("Could not found two parallel lines")

    if debug:
        x1, y1 = parallel_lines[0]["p1"]
        x2, y2 = parallel_lines[0]["p2"]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        x1, y1 = parallel_lines[1]["p1"]
        x2, y2 = parallel_lines[1]["p2"]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    x, y = intersect(*parallel_lines)

    return int(x), int(y)


if __name__ == "__main__":
    # img = cv2.imread("samples/12.jpg")
    img = cv2.imread("samples/4.jpg")
    vp = get_vanishing_point(img)
    print(vp)
    cv2.imshow("frame", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
