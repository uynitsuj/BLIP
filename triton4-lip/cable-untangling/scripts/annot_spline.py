import cv2
import numpy as np
import os
import math
import time

class KeypointsAnnotator:
    def __init__(self):
        pass

    def load_image(self, img):
        self.img = img
        self.eroded = cv2.erode(self.img, np.ones((2,2), np.uint8), iterations=1)
        self.white_points = np.where(self.eroded[:800, ...] > 100)
        self.click_to_kpt = {0:"PULL1", 1:"PULL2"}
        self.drawing = False
    
    def project_point_onto_cable(self, point):
        closest_point = np.argmin(np.linalg.norm(np.array([[point[0]], [point[1]]]) - np.array([self.white_points[1], self.white_points[0]]), axis=0))
        return self.white_points[1][closest_point], self.white_points[0][closest_point]

    def mouse_callback(self, event, x, y, flags, param):
        cv2.imshow("pixel_selector", self.img)
        # project the point onto the image mask
        x, y = self.project_point_onto_cable([x, y])
        
        # click and drag (hold mouse down)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = not self.drawing
            time.sleep(0.3)

        # if event == cv2.EVENT_LBUTTONUP:
        #     self.drawing = False

        if event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                if len(self.clicks) == 0 or np.linalg.norm(np.array([x, y]) - np.array(self.clicks[-1])) > 5:
                    prev_point_avgd = None
                    if len(self.clicks) > 1 and np.linalg.norm(np.array([x, y]) - np.array(self.clicks[-2])) < 10:
                        prev_point_avgd = (np.array([x, y]) + np.array(self.clicks[-2])) / 2
                    elif len(self.clicks) > 0:
                        prev_point_avgd = np.array(self.clicks[-1])
                    if prev_point_avgd is not None:
                        px, py = self.project_point_onto_cable(prev_point_avgd)
                        self.filtered_clicks.append([px, py])

                    # if len(self.filtered_clicks) > 1:
                    #     cv2.line(self.img, (self.filtered_clicks[-2][0], self.filtered_clicks[-2][1]), (self.filtered_clicks[-1][0], self.filtered_clicks[-1][1]), (0, 0, 255), 1)
                    if len(self.clicks) > 1:
                        cv2.line(self.img, (self.clicks[-2][0], self.clicks[-2][1]), (self.clicks[-1][0], self.clicks[-1][1]), (0, 0, 255), 1)
                    self.clicks.append([x, y])

                    print("Clicked at: ", x, y)
                    cv2.circle(self.img, (x, y), 1, (255, 0, 0), -1)
                    cv2.imshow("pixel_selector", self.img)

    def run(self, img):
        self.load_image(img.copy())
        self.total_clicks = []
        self.total_filtered_clicks = []
        self.clicks = []
        self.filtered_clicks = []
        cv2.namedWindow('pixel_selector')#, cv2.WINDOW_NORMAL)
        cv2.resizeWindow('pixel_selector', 600, 600) 
        cv2.setMouseCallback('pixel_selector', self.mouse_callback)
        cv2.imshow("pixel_selector", self.img)
        while True:
            waitkey = cv2.waitKey(33)
            if waitkey & 0xFF == 27:
                break
            if waitkey == ord('r'):
                self.total_clicks = []
                self.clicks = []
                self.filtered_clicks = []
                self.load_image(img.copy())
                cv2.imshow("pixel_selector", self.img)
                print('Erased annotations for current image')
            if waitkey == ord('s'):
                self.total_clicks.append(self.clicks)
                self.total_filtered_clicks.append(self.filtered_clicks)
                self.clicks = []
                self.filtered_clicks = []
                print('Starting new spline')
            if waitkey == ord('z'):
                self.clicks = self.clicks[:-1]
                self.filtered_clicks = self.filtered_clicks[:-1]
                self.load_image(img.copy())
                print(len(self.clicks))
                # for i in range(len(self.filtered_clicks) - 1):
                #     cv2.line(self.img, (self.filtered_clicks[i][0], self.filtered_clicks[i][1]), (self.filtered_clicks[i+1][0], self.filtered_clicks[i+1][1]), (0, 0, 255), 1)
                for i in range(len(self.filtered_clicks) - 1):
                    cv2.line(self.img, (self.filtered_clicks[i][0], self.filtered_clicks[i][1]), (self.filtered_clicks[i+1][0], self.filtered_clicks[i+1][1]), (0, 0, 255), 1)
                for i in range(len(self.clicks)):
                    cv2.circle(self.img, (self.clicks[i][0], self.clicks[i][1]), 1, (255, 0, 0), -1)
                cv2.imshow("pixel_selector", self.img)
                print('Erased last annotation for current image')

        cv2.destroyAllWindows()
        if (len(self.clicks) > 0):
            self.total_clicks.append(self.clicks)
            self.total_filtered_clicks.append(self.filtered_clicks)

        print('tot clicks', self.total_clicks)
        print('tot filtered clicks', self.total_filtered_clicks)
        return self.total_clicks, self.total_filtered_clicks

if __name__ == '__main__':
    pixel_selector = KeypointsAnnotator()

    image_dir = '/Users/vainaviv/Documents/GitHub/untangling_long_cables/raw_data/generalize_knots_dataset/fig8_overhand/test'

    # image_dir = 'single_knots' # Should have images like 00000.jpg, 00001.jpg, ...
    output_dir = 'fig8_overhand_test' # Will have real_data/images and real_data/annots
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    keypoints_output_dir = os.path.join(output_dir, 'annots')
    images_output_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(keypoints_output_dir):
        os.mkdir(keypoints_output_dir)
    if not os.path.exists(images_output_dir):
        os.mkdir(images_output_dir)

    for i,f in enumerate(sorted(os.listdir(image_dir))[::-1]):
        if 'depth' in f:
            continue
        print("Img %d"%i)
        image_path = os.path.join(image_dir, f)
        img = cv2.imread(image_path)
        # pixel_selector.load_image(img.copy())
        # continue
        #assert(img.shape[0] == img.shape[1] == 60)
        # image_outpath = os.path.join(images_output_dir, '%05d.png'%fnumber)
        keypoints_outpath = os.path.join(keypoints_output_dir, f[:-4] + '.npy')
        if os.path.exists(keypoints_outpath):
            print("Already exists: ", keypoints_outpath)
            continue
        # cv2.imwrite(image_outpath, img)
        annots, filtered_annots = pixel_selector.run(img)
        print("---")
        # annots = np.array(annots)
        np.save(keypoints_outpath, {'points': annots, 'filtered_points': filtered_annots})
