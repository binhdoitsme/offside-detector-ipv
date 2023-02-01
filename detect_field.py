import cv2
import numpy as np
import os


def histeq(img: np.ndarray):
    """Equalize multi-channel histogram"""
    _img = img.copy()
    for i, _ in enumerate(img):
        _img[i] = cv2.equalizeHist(img[i])
    return _img


def get_field_mask(img: cv2.Mat, debug=False):
    """Get the mask of the field"""
    low_threshold = np.array([0, 90, 0], dtype=np.uint8)
    high_threshold = np.array([170, 255, 255], dtype=np.uint8)

    img = histeq(img)
    mask: cv2.Mat = cv2.inRange(img, low_threshold, high_threshold)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # morph open -> remove small links between inner-field and outer-field
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=15)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=25)

    # create a black frame border of 1px wide to have a closed field contour
    mask[0, :] = 0
    mask[:, 0] = 0
    mask[-1, :] = 0
    mask[:, -1] = 0

    if debug:
        cv2.imshow("mask", mask)

    contours, _ = cv2.findContours(
        mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    final_mask = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(final_mask, [contours[0]], 0, (255,255,255), -1)
    return cv2.Mat(final_mask)


if __name__ == "__main__":
    base_folder = "samples"
    sample_files = os.listdir(base_folder)

    for filename in sample_files[:2]:
        img = cv2.imread(os.path.join(base_folder, filename))
        field_mask = get_field_mask(img)
        # cv2.drawContours(img, [field_contour], -1, (255, 255, 0), 3)
        result = cv2.bitwise_and(img, img, mask=field_mask)
        cv2.imshow(filename, result)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
