from yumirws.yumi import YuMi
from yumiplanning.yumi_kinematics import YuMiKinematics as YK
from untangling.utils.tcps import ABB_WHITE

# import time; time.sleep(10)

L_TCP=ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame)
R_TCP=ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame)

yumi = YuMi(l_tcp=L_TCP, r_tcp=R_TCP)
print('right: ', repr(yumi.right.get_joints()))
print('left: ', repr(yumi.left.get_joints()))