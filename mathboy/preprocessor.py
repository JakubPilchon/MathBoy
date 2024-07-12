import cv2 as cv
import os

class Preprocessor:
    def open_img(self):
        # check if file exists 
        assert os.path.isfile('img.jpg')

        img = cv.imread('img.jpg')

        print(img.shape)

        img_pre = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, img_pre = cv.threshold(img_pre, 240, 255, cv.THRESH_BINARY_INV)

        img_pre = img_pre[10:694, 10:1244]
        #img_pre = cv.bitwise_not(img_pre)
        dilated = cv.dilate(img_pre, cv.getStructuringElement(cv.MORPH_RECT,(5,5)), iterations=5)

        contours, hierarchy = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        cv.drawContours(img_pre, contours, -1, (0,255,0), 3)



        for n in contours:
            x,y,w,h = cv.boundingRect(n)
            if h < 40: h = 40

            if w >= 80:
                cv.rectangle(img, (x,y), (x+w, y+h), (0,255,00), 5)

        cv.imshow("img_pre",img_pre)
        cv.imshow("img", img)
        k = cv.waitKey(0)

if __name__ == "__main__":
    pre = Preprocessor()
    pre.open_img()