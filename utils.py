import pydrake
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.math import RigidTransform


def CreateIiwaPlant():
    plant = MultibodyPlant(1e-3)
    parser = Parser(plant=plant)

    iiwa_drake_path = (
        "drake/manipulation/models/iiwa_description/iiwa7/iiwa7_no_collision.sdf")
    iiwa_path = pydrake.common.FindResourceOrThrow(iiwa_drake_path)
    robot_model = parser.AddModelFromFile(iiwa_path)

    # weld robot to world frame.
    plant.WeldFrames(A=plant.world_frame(),
                     B=plant.GetFrameByName("iiwa_link_0"),
                     X_AB=RigidTransform.Identity())
    plant.Finalize()

    # store reference to all link frames
    link_frames = []
    for i in range(8):
        link_frames.append(plant.GetFrameByName("iiwa_link_%d" % i))

    return plant, link_frames
