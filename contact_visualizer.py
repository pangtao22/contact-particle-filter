import os

import meshcat
import numpy as np
import yaml
from drake import (lcmt_iiwa_status, lcmt_contact_info,)
from pydrake.common.value import AbstractValue
from pydrake.math import RollPitchYaw
from pydrake.systems.framework import (
    LeafSystem, PublishEvent, TriggerType,
)
from pydrake.systems.meshcat_visualizer import AddTriad

from contact_particle_filter.utils import *


def DrawArrow(vis, name, origin, direction,
              length=0.1,
              radius=0.0025,
              prefix="normals",
              color=0xff0000,
              opacity=1.):
    R = np.zeros((3, 3))
    magnitude = np.linalg.norm(direction)
    y = direction / magnitude
    R[:, 1] = y
    if np.max(np.abs(y[1:])) < 1e-3:
        R[:, 0] = [-y[1], y[0], 0]
    else:
        R[:, 0] = [0, -y[2], y[1]]
    R[:, 0] /= np.sqrt((R[:, 0] ** 2).sum())
    R[:, 2] = np.cross(R[:, 0], y)
    X_WC = np.eye(4)
    X_WC[0:3, 0:3] = R
    X_WC[0:3, 3] = origin
    # Scale cylinder
    T_scale = meshcat.transformations.translation_matrix(
        [0, length / 2, 0])
    T_scale[1, 1] = length
    # - "expand" cylinders to a visible size.
    T_scale[0, 0] *= radius
    T_scale[2, 2] *= radius
    # Publish.
    vis[prefix][name].set_object(
        meshcat.geometry.Cylinder(height=1, radius=1),
        meshcat.geometry.MeshLambertMaterial(color=color, opacity=opacity))
    vis[prefix][name].set_transform(X_WC.dot(T_scale))


# plot normals
def DrawNormals(vis, points, indices, face_normals, X_WB):
    for i, point in enumerate(points):
        DrawArrow(vis, "arrow%d" % i,
                  origin=X_WB.multiply(point),
                  direction=X_WB.rotation().dot(face_normals[indices[i]]),
                  length=0.05)


def DrawPointCloud(vis, name, points, color, X_WB, size=0.005):
    n = len(points)
    colors = np.zeros((3, n))
    colors[0] = color[0]
    colors[1] = color[1]
    colors[2] = color[2]

    vis[name].set_object(
        meshcat.geometry.PointCloud(position=points.T, color=colors, size=size))
    vis[name].set_transform(X_WB)


def DrawRobot(vis, pose_bundle, X_WB_list, n=8):
    # load yaml configuration file
    pkg_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(pkg_path, "cpf_config.yaml"), 'r') as stream:
        config = yaml.safe_load(stream)

    # visualize meshes in Meshcat
    mesh_relative_paths = config["mesh_relative_paths"]
    vis.delete()
    for i in range(n):
        mesh_path = pkg_path + mesh_relative_paths[i]
        vis["link_%d" % i].set_object(
            meshcat.geometry.ObjMeshGeometry.from_file(mesh_path))

    for i in range(n):
        vis["link_%d" % i].set_transform(
            pose_bundle[i].multiply(X_WB_list[i]).matrix())


# visualize a facet
def DrawFacet(vis, abcd, name, center=None,
              prefix='facets', radius=0.02, thickness=0.001, color=0xffffff,
              opacity=0.6):
    normal = np.array(abcd[:3]).astype(float)
    normal /= np.linalg.norm(normal)
    d = -abcd[3] / np.linalg.norm(normal)

    R = np.eye(3)
    R[:, 2] = normal
    z = normal
    if abs(z[0]) < 1e-8:
        x = np.array([0, -normal[2], normal[1]])
    else:
        x = np.array([-normal[1], normal[0], 0])
    x /= np.linalg.norm(x)
    R[:, 0] = x
    R[:, 1] = np.cross(z, x)

    X = np.eye(4)
    Rz = RollPitchYaw(np.pi / 2, 0, 0).ToRotationMatrix().matrix()
    X[:3, :3] = R.dot(Rz)
    if center is None:
        X[:3, 3] = d * normal
    else:
        X[:3, 3] = center

    X_normal = X.copy()
    X_normal[:3, :3] = R

    material = meshcat.geometry.MeshLambertMaterial(
        color=color, opacity=opacity)

    vis[prefix][name]["plane"].set_object(
        meshcat.geometry.Cylinder(thickness, radius), material)

    normal_vertices = np.array([[0, 0, 0], [0, 0, radius]]).astype(float)
    vis[prefix][name]["normal"].set_object(
        meshcat.geometry.Line(
            meshcat.geometry.PointsGeometry(normal_vertices.T)))

    vis[prefix][name]["plane"].set_transform(X)
    vis[prefix][name]["normal"].set_transform(X_normal)


def DrawFacets(vis: meshcat.Visualizer, name: str, normals, centers, radius):
    assert len(normals) == len(centers)
    prefix = name + "_facets"
    vis[prefix].delete()
    for i in range(len(normals)):
        plane_equation = np.hstack((normals[i], [0]))
        DrawFacet(vis, plane_equation, center=centers[i],
                  name=str(i), radius=radius, prefix=prefix)


class IiwaContactVisualizer(LeafSystem):
    def __init__(self, drake_lcm, contact_discrimination=False):
        LeafSystem.__init__(self)
        self.drake_lcm = drake_lcm
        self.set_name('iiwa_cpf_visualizer')
        self.DeclarePeriodicPublish(1. / 30, 0.0)  # draw at 30fps

        self.DeclareAbstractInputPort("iiwa_status",
                                      AbstractValue.Make(lcmt_iiwa_status()))

        self.DeclareAbstractInputPort("contact_postion",
                                      AbstractValue.Make(lcmt_contact_info()))
        # self.DeclareAbstractInputPort('contact_discrimination',
        #     AbstractValue.Make(lcmt_contact_discrimination()))

        self.vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        # self.vis = meshcat.Visualizer()
        self.plant, self.link_frames = CreateIiwaPlant()
        self.context = self.plant.CreateDefaultContext()
        self.x = np.zeros(
            self.plant.num_velocities() + self.plant.num_positions())
        self.pose_bundle = [None] * len(self.link_frames)
        # relative transfrom between mesh frame (B) and link frame (L).
        self.X_LB_list = []

        def on_initialize(context, event):
            self.load()

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize))

        self.t_last_print = -np.inf

    def load(self):
        """
        Loads robot meshes into meshcat
        :return:
        """
        n = 8

        rgb_grey = 0xa1a1a1
        rgb_orange = 0xff7800
        rgb_ee = 0xe3dac9

        colors = [rgb_grey, rgb_grey, rgb_orange, rgb_grey, rgb_orange,
                  rgb_grey, rgb_orange, rgb_ee]

        # load yaml configuration file
        pkg_path = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(pkg_path, "cpf_config.yaml"), 'r') as stream:
            config = yaml.safe_load(stream)

        # visualize meshes in Meshcat
        mesh_relative_paths = config["mesh_relative_paths"]
        self.vis.delete()
        for i in range(n):
            mesh_path = pkg_path + mesh_relative_paths[i]
            mesh_material = meshcat.geometry.MeshLambertMaterial(
                color=colors[i], opacity=0.8)
            self.vis["link_%d" % i].set_object(
                meshcat.geometry.ObjMeshGeometry.from_file(mesh_path),
                mesh_material)

        # Add coordinate frame to link 6
        # AddTriad(self.vis, "link6_body_frame", "link_6", length=0.15,
        #          radius=0.006)

        # set relative mesh transforms
        for i in range(n):
            X_LB = RigidTransform.Identity()
            X_LB.set_translation(config["mesh_translation"][i])
            self.X_LB_list.append(X_LB)

        # draw table
        material = meshcat.geometry.MeshLambertMaterial(
            color=0x90ee90, opacity=0.7)

        self.vis["table"].set_object(
            meshcat.geometry.Box([1.2, 0.76, 10]), material)
        dx_table_center_to_robot_base = 0.3257
        dz_table_top_robot_base = 0.0127

        X_WT = meshcat.transformations.translation_matrix(
            [dx_table_center_to_robot_base, 0, -5 - dz_table_top_robot_base])
        self.vis["table"].set_transform(X_WT)

    def CalcLinkPoses(self, q):
        self.x[:self.plant.num_positions()] = q
        self.context.SetDiscreteState(self.x)
        for i, frame in enumerate(self.link_frames):
            X_WL = self.plant.CalcRelativeTransform(
                self.context,
                frame_A=self.plant.world_frame(),
                frame_B=frame)
            X_WB = X_WL.multiply(self.X_LB_list[i])
            self.pose_bundle[i] = X_WB

    def UpdateLinkPoses(self, q):
        self.CalcLinkPoses(q)
        for i in range(len(self.link_frames)):
            self.vis["link_%d" % i].set_transform(self.pose_bundle[i].matrix())

    def DrawForce(self, p_LC_L, link_idx, f_W, name, color=0xff0000):
        p_WC_W = self.pose_bundle[link_idx].multiply(p_LC_L)
        DrawArrow(
            self.vis,
            name=name,
            length=0.15,
            radius=0.01,
            origin=p_WC_W,
            direction=f_W,
            prefix="contact_forces",
            color=color)

    def DoPublish(self, context, event):
        LeafSystem.DoPublish(self, context, event)
        self.drake_lcm.HandleSubscriptions(0)
        status_msg = self.EvalAbstractInput(context, 0).get_value()
        contact_msg = self.EvalAbstractInput(context, 1).get_value()

        q = status_msg.joint_position_measured
        if len(q) != 7:
            return

        # draw the robot
        self.UpdateLinkPoses(q)

        # print the pose of link 7 every second.
        t = context.get_time()
        if t - self.t_last_print > np.inf:
            self.t_last_print = t
            rpy = RollPitchYaw(self.pose_bundle[7].rotation())
            print(t, "link 7 position: ",
                  self.pose_bundle[7].multiply(np.array([0, 0, 0.0])),
                  " rpy: ", rpy.roll_angle(), rpy.pitch_angle(),
                  rpy.yaw_angle())

        # draw contact forces
        num_contacts = contact_msg.num_contacts
        if num_contacts > 0:
            for i, link_idx in enumerate(contact_msg.link_indices):
                position = contact_msg.position[i]
                f_W = np.array(contact_msg.normal[i])
                p_LC_L = self.X_LB_list[link_idx].inverse().multiply(position)
                self.DrawForce(p_LC_L, link_idx, -f_W, name="detected")

        else:
            self.vis["contact_forces"].delete()


class CpfMonteCarloTestResultVisualizer:
    def __init__(self):
        cpf_viz = IiwaContactVisualizer(drake_lcm=None)
        cpf_viz.load()
        self.cpf_viz = cpf_viz

    def DrawTestResult(self, test_idx, test_result):
        q = test_result["joint_angles"][test_idx]
        self.cpf_viz.UpdateLinkPoses(q)
        force_scale = 50

        for i in range(test_result["true_num_contacts"]):
            link_idx_true = test_result["true_contact_link"][test_idx][i]
            f_W_true = test_result["true_contact_force_f_W"][test_idx][i]
            p_LC_L_true = test_result["true_contact_point_p_LC_L"][test_idx][i]
            p_WC_W_true = self.cpf_viz.pose_bundle[link_idx_true].multiply(
                p_LC_L_true)
            DrawArrow(self.cpf_viz.vis, name="true_{}".format(i),
                      origin=p_WC_W_true, direction=-f_W_true,
                      length=np.linalg.norm(f_W_true) / force_scale,
                      prefix="contact_forces", color=0x00ff00, opacity=0.8)

        if test_result["detected_num_contacts"][test_idx] > 0:
            link_idx_detected = int(test_result[
                                        "detected_contact_link"][test_idx])
            f_W_detected = test_result[
                "detected_contact_force_f_W_mean"][test_idx]
            p_LC_L_detected = test_result[
                "detected_contact_point_p_LC_L_mean"][test_idx]
            p_WC_W_detected = self.cpf_viz.pose_bundle[
                link_idx_detected].multiply(p_LC_L_detected)
            DrawArrow(self.cpf_viz.vis, name="detected",
                      origin=p_WC_W_detected, direction=-f_W_detected,
                      length=np.linalg.norm(f_W_detected) / force_scale,
                      prefix="contact_forces", color=0xff0000, opacity=0.8)

            min_cost = np.min(
                test_result["optimal_particle_QP_values"][test_idx])
            max_cost = np.max(
                test_result["optimal_particle_QP_values"][test_idx])

            print("max_min_cost: ", max_cost, min_cost)
