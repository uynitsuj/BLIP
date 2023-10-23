
from detectron2_repo import analysis as loop_detectron

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def closest_valid_point(color, yx):
    valid_y, valid_x = np.nonzero((color[:, :, 0] > 100))
    pts = np.vstack((valid_y, valid_x)).T
    return pts[np.argmin(np.linalg.norm(pts - np.array(yx)[None, :], axis=-1))]

def get_endpoints(img):
    # model not used, already specified in loop_detectron
    endpoint_boxes, out_viz = loop_detectron.predict(img, thresh=0.99, endpoints=True)
    endpoints = []
    
    for box in endpoint_boxes:
        xmin, ymin, xmax, ymax = box
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        new_yx = closest_valid_point(img, np.array([y, x]))
        endpoints.append([new_yx, new_yx])
    
    endpoints = np.array(endpoints).astype(np.int32)
    
    
    endpoints = endpoints.astype(np.int32).reshape(-1, 2, 2)
    endpoints = endpoints[:, 0, :]  # This line extracts the first point from each endpoint pair
    
    return endpoints


if __name__ == '__main__':
    image_path = './cable_insert_imgs/straight6.png'
    img = np.asarray(Image.open(image_path))

    endpoints = get_endpoints(img)
    print(endpoints)
     # Visualization
    plt.clf()
    plt.title("Endpoint detections visualization")
    
    if len(endpoints) > 0:
        plt.scatter(endpoints[:, 1], endpoints[:, 0])

    plt.imshow(img)
    plt.axis('off')
    plt.show()