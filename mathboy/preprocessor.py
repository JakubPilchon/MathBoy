import cv2 as cv
import os

class Preprocessor:
    def open_img(self):
        # check if file exists 
        assert os.path.isfile('img.jpg')

        img = cv.imread('img.jpg')

        img_pre = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #img = cv.THRESH_BINARY_INV()
        _, img_pre = cv.threshold(img_pre, 240, 255, cv.THRESH_BINARY_INV)

        contours, hierarchy = cv.findContours(img_pre, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        cv.drawContours(img_pre, contours, -1, (0,255,0), 3)

        contours = contours[0].reshape(-1, 2)

        
        
        for (x, y) in contours:
            cv.circle(img_pre, (x, y), 100, (0, 255,0), 3)

        print(contours)

        cv.imshow("img_pre",img_pre)
        cv.imshow("img", img)
        k = cv.waitKey(0)


if __name__ == "__main__":
    pre = Preprocessor()
    pre.open_img()