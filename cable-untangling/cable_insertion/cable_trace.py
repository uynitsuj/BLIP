
from detectron2_repo import analysis as loop_detectron
from tusk_pipeline.tracer import Tracer, AnalyticTracer
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


    tracer = Tracer()
    analytic_tracer = AnalyticTracer()
    eval_folder = '../data/real_data/real_data_for_tracer/test'

    thresh_img = np.where(img[:,:,:3] > 100, 255, 0).astype('uint8')
    start_pixels = endpoints[0]
    print(start_pixels)
    start_pixels, _ = analytic_tracer.trace(thresh_img, start_pixels, path_len=6, viz=False, idx=0)
    if len(start_pixels) < 5:
        exit(0)

    spline = tracer.trace(img, start_pixels, path_len=200, viz=True, mask=False)
    spline = tracer.trace(thresh_img, start_pixels, path_len=200, viz=True, mask=True)

    print(spline)