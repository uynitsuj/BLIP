from scripts.full_pipeline_trunk import FullPipeline
import argparse
import logging
import numpy as np
import os


class MultiCable(FullPipeline):
    def __init__(self, viz, loglevel, initialize_iface):
        FullPipeline.__init__(self, viz, loglevel, initialize_iface)
        self.output_vis_dir = "./multicable/"
        self.iface.home()
        self.iface.sync()
        self.img = self.iface.take_image()

    def get_analytic_trace(self):
        analytic_trace_problem = True
        while analytic_trace_problem:
            self.get_endpoints_from_clicks(take_image=False)
            endpoint = self.endpoints[0]
            starting_pixels, analytic_trace_problem = self.get_trace_from_endpoint(endpoint)
        return starting_pixels

    def do_learned_perception_pipeline(self, starting_pixels):
        # get trace and u/o crossings
        self.tkd._set_data(self.img.color._data, starting_pixels)
        perception_result = self.tkd.perception_pipeline(endpoints=self.endpoints, viz=False, vis_dir=self.output_vis_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help="Print more statements",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.INFO,
    )

    args = parser.parse_args()
    logLevel = args.loglevel


    input_dir = "rope_knot_images/knot_challenging_distractors/step_1/"
    output_dir = "rope_knot_images_analytic_traces/knot_challenging_distractors/step_1"
    fullPipeline = MultiCable(viz=False, loglevel=logLevel, initialize_iface=True)
    for file in os.listdir(input_dir):
        if file.endswith(".npy"):
            img = np.load(f"{input_dir}/{file}", allow_pickle=True).item()
            print(file)
            fullPipeline.img = img
            starting_pixels = fullPipeline.get_analytic_trace()
            print("starting_pixels", starting_pixels)
            np.save(f"{output_dir}/starting_pixels_{file}", starting_pixels)
            test = np.load(f"{output_dir}/starting_pixels_{file}", allow_pickle=True)
            print("TEST", test)