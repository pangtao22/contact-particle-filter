import time

import numpy as np

import lcm
from contact_particle_filter.contact_particle_filter_core import ContactParticleFilter
from drake import lcmt_contact_info, lcmt_iiwa_status

if __name__ == "__main__":
    cpf = ContactParticleFilter()

    def HandleIiwaStatusMessage(channel, data):
        msg = lcmt_iiwa_status.decode(data)
        q = msg.joint_position_measured
        tau_external = np.array(msg.joint_torque_external)

        cpf.RunContactParticleFilter(q, tau_external)

    lc = lcm.LCM()
    subscription = lc.subscribe("IIWA_STATUS", HandleIiwaStatusMessage)
    subscription.set_queue_capacity(1)

    try:
        while True:
            lc.handle()
            result = cpf.CalcBelief()
            num_contacts = result[0]
            msg_to_send = lcmt_contact_info()
            msg_to_send.timestamp = int(time.time() * 1e6)
            msg_to_send.num_contacts = num_contacts
            if num_contacts > 0:
                msg_to_send.position = result[1]
                msg_to_send.normal = result[2]
                msg_to_send.link_indices = result[3]
            else:
                time.sleep(0.008)

            if msg_to_send.num_contacts != len(msg_to_send.link_indices):
                print("why???")
            lc.publish("CONTACT_INFO", msg_to_send.encode())
    except KeyboardInterrupt:
        pass

