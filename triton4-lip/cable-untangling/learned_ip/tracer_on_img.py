import sys
# add paths (clean up later??)
sys.path.insert(0, '../')
sys.path.insert(0, '../..')
sys.path.insert(0, '../scripts/') 
sys.path.insert(0, '~')
sys.path.insert(0, '../../autolab_core/')
sys.path.insert(0, '../../detectron2_repo/')
sys.path.insert(0, '../../yumiplanning')
sys.path.insert(0, '../../yumirws')
sys.path.insert(0, '../../tracikpy/') 
sys.path.insert(0, '../../tracikpy/tracikpy/') 
sys.path.insert(0, '../../phoxipy/') 
sys.path.insert(0, '/home/mallika/triton4-lip/yumiplanning/') 
sys.path.insert(0, '/home/mallika/triton4-lip/yumiplanning/yumiplanning/') 

sys.path.insert(0, '../untangling/utils/cable_tracing/')


# from full_pipeline_trunk import FullPipeline
from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
import numpy as np
import time


if __name__ == "__main__":
    SPEED = (0.4, 6 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )
    iface.open_grippers()
    time.sleep(10)
    iface.close_grippers()
    iface.sync()
