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
    iface.sync()
    iface.close_grippers()
    iface.sync()
