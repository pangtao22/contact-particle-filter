from pydrake.lcm import DrakeLcm

from pydrake.systems.lcm import LcmSubscriberSystem, LcmInterfaceSystem
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from contact_particle_filter.contact_visualizer import *

from plan_runner.plan_utils import RenderSystemWithGraphviz

if __name__ == "__main__":
    drake_lcm = DrakeLcm()

    builder = DiagramBuilder()
    cpf_vis = builder.AddSystem(IiwaContactVisualizer(drake_lcm))
    builder.AddSystem(LcmInterfaceSystem(drake_lcm))

    iiwa_lcm_sub = builder.AddSystem(LcmSubscriberSystem.Make(
        channel="IIWA_STATUS", lcm_type=lcmt_iiwa_status, lcm=drake_lcm))
    builder.Connect(iiwa_lcm_sub.get_output_port(0),
                    cpf_vis.get_input_port(0))

    contact_lcm_sub = builder.AddSystem(LcmSubscriberSystem.Make(
        channel="CONTACT_INFO", lcm_type=lcmt_contact_info, lcm=drake_lcm))
    builder.Connect(contact_lcm_sub.get_output_port(0),
                    cpf_vis.get_input_port(1))

    diagram = builder.Build()
    # RenderSystemWithGraphviz(diagram)

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)

    simulator.AdvanceTo(np.inf)




