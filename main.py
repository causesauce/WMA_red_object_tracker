import cv2 as cv
import numpy as np
import argparse


def main(args):
    video = cv.VideoCapture(args.input_video)
    ret, frame = video.read()
    lower_bound_red, upper_bound_red = (np.array([160, 50, 50]), np.array([180, 255, 255]))
    x_center_frame = int(video.get(cv.CAP_PROP_FRAME_WIDTH)) // 2

    while ret:
        blurFrame = cv.GaussianBlur(frame, (11, 11), 0)
        hsv = cv.cvtColor(blurFrame, cv.COLOR_BGR2HSV)

        mask = cv.inRange(hsv, lower_bound_red, upper_bound_red)

        kernel = np.ones((13, 13), np.uint8)
        opened = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)


        contours, _ = cv.findContours(closed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        max_contour = contours[0]
        for contour in contours:
            if cv.contourArea(contour) > cv.contourArea(max_contour):
                max_contour = contour

        contour = max_contour
        approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)

        x, y, w, h = cv.boundingRect(approx)
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        M = cv.moments(contour)
        cx = int(M['m10'] // M['m00'])
        cy = int(M['m01'] // M['m00'])
        cv.circle(frame, (cx, cy), (h // 2 + w // 2) // 2, (0, 255, 0), 2)

        x_end = cx - x_center_frame
        cv.line(frame, (x_center_frame, 50), (x_center_frame + x_end, 50), (0, 255, 0), 3)
        cv.circle(frame, (x_center_frame, 50), 3, (0, 0, 0), -1)

        cv.imshow("frame", frame)
        cv.imshow("opened", opened)

        key = cv.waitKey(10)
        if key == ord('q'):
            break
        ret, frame = video.read()

    video.release()
    cv.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser(description=('This script divides video'
                                                  'stream into R,G,B channels'))

    parser.add_argument('-i',
                        '--input_video',
                        type=str,
                        required=True,
                        help='A video file that will be processed')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
