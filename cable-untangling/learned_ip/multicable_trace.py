from scripts.full_pipeline_trunk import FullPipeline
import argparse
import logging
import numpy as np





class MultiCable(FullPipeline):
    def __init__(self, viz, loglevel, initialize_iface):
        FullPipeline.__init__(self, viz, loglevel, initialize_iface)
        self.output_vis_dir = "./multicable/"
        self.img = self.iface.take_image()

    def get_analytic_trace(self):
        analytic_trace_problem = True
        while analytic_trace_problem:
            self.get_endpoints_from_clicks()
            # trying each endpoint
            for endpoint in self.endpoints:
                print("endpoint", endpoint)
                starting_pixels, analytic_trace_problem = self.get_trace_from_endpoint(endpoint)
                starting_pixels = np.array(starting_pixels)
                if not analytic_trace_problem:
                    break
            # if both fail, try endpoint clicks again
        return starting_pixels


    def do_learned_perception_pipeline(self, starting_pixels):
        # get trace and u/o crossings
        self.tkd._set_data(self.img.color._data, starting_pixels)
        perception_result = self.tkd.perception_pipeline(endpoints=self.endpoints, viz=True, vis_dir=self.output_vis_dir)


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

    fullPipeline = MultiCable(viz=False, loglevel=logLevel, initialize_iface=True)
    starting_pixels = fullPipeline.get_analytic_trace()
    fullPipeline.do_learned_perception_pipeline(starting_pixels)



    
    
    