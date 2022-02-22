import time

import lcm
import numpy as np

from drake import lcmt_iiwa_status

if __name__ == "__main__":
    tau_external = np.array([0., -4.88641225e+00, 0.,  3.36167658e+00, 0., 0., 0.])
    q0 = [0, 0, 0, -1.75, 0, 1.0, 0]
    nq = 7

    # message with contact
    msg = lcmt_iiwa_status()
    msg.num_joints = nq
    msg.joint_position_measured = q0
    msg.joint_torque_external = tau_external
    msg.joint_velocity_estimated = np.zeros(nq)
    msg.joint_position_ipo = np.zeros(nq)
    msg.joint_torque_measured = np.zeros(nq)
    msg.joint_torque_commanded = np.zeros(nq)
    msg.joint_position_commanded = np.zeros(nq)

    # message without contact
    msg2 = lcmt_iiwa_status()
    msg2.num_joints = nq
    msg2.joint_position_measured = q0
    msg2.joint_torque_external = np.zeros(nq)
    msg2.joint_velocity_estimated = np.zeros(nq)
    msg2.joint_position_ipo = np.zeros(nq)
    msg2.joint_torque_measured = np.zeros(nq)
    msg2.joint_torque_commanded = np.zeros(nq)
    msg2.joint_position_commanded = np.zeros(nq)

    lc = lcm.LCM()

    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time % 5 < 4:
            lc.publish("IIWA_STATUS", msg.encode())
        else:
            lc.publish("IIWA_STATUS", msg2.encode())
        time.sleep(0.004)