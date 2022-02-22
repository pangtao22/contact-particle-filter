import os

import trimesh
import yaml
import numpy as np
import scipy.sparse as sparse

from pydrake.multibody.tree import JacobianWrtVariable

from contact_particle_filter.utils_cython import *
from contact_particle_filter.utils import *
from plan_runner.low_pass_filter import LowPassFilter

import contact_particle_filter.emosqp as emosqp


class ContactParticleFilter:
    """
    Many parameters are hard-coded for iiwa7.
    """
    def __init__(self):
        # create multibodyplant
        self.plant, self.link_frames = CreateIiwaPlant()
        self.nq = self.plant.num_positions()
        # the robot has 8 links. The -1 excludes the world body.
        self.nb = self.plant.num_bodies() - 1

        # create a context and a mutable reference to the state vector.
        self.context = self.plant.CreateDefaultContext()
        self.x = np.zeros(self.plant.num_positions() + self.plant.num_velocities())
        self.pose_bundle = [None] * self.nb

        # load yaml configuration file
        pkg_path = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(pkg_path, "cpf_config.yaml"), 'r') as stream:
            config = yaml.safe_load(stream)
        self.config = config

        # parameters of contact particle filter
        self.kp = config["number_of_particles"]  # number of particles
        self.nd = 4  # number of friction cone rays
        self.friction_coefficient = config["friction_coefficient"]
        # a particle is a point in a link's local frame.
        self.particles = np.zeros((self.kp, 3))
        self.f_W_particles = np.zeros_like(self.particles)
        self.particles_optimal_values = np.full(self.kp, -np.inf)
        # the mesh triangle indices of particles.
        self.indices_in_mesh = np.zeros(self.kp, dtype=int)
        self.has_contact = False

        # get mesh paths
        # links from which particles are sampled.
        self.active_link_idx = config["active_link_idx"]
        # assert self.active_link_idx == [6, 7]
        mesh_list = []
        face_normals_list = []
        link_areas = np.zeros(len(self.active_link_idx))  # used as sampling weights.
        mesh_relative_paths = config["mesh_relative_paths"]
        for j, i in enumerate(self.active_link_idx):
            mesh_path = pkg_path + mesh_relative_paths[i]
            mesh = trimesh.load(mesh_path)
            mesh_list.append(mesh)
            face_normals_list.append(mesh.face_normals.copy())
            link_areas[j] = mesh.area

        self.num_link_samples = np.round(link_areas/link_areas.sum()*self.kp).astype(int)
        diff = self.kp - self.num_link_samples.sum()
        self.num_link_samples[0] += diff
        assert self.num_link_samples.sum() == self.kp

        self.mesh_list = mesh_list
        self.face_normals_list = face_normals_list
        self.link_areas = link_areas

        # set mesh relative transforms.
        # The mesh of a link used for contact detection could be a rigid transform away
        # from the body frame of the link. At the moment these transforms are
        # hard-coded into the following list.

        # frame relationship:
        # W: world frame.
        # -- L: link frame, body frame of links in MultibodyPlant.
        # ---- B: mesh frame, X_LB are defined in the yaml configuration file.
        self.X_LB_list = []
        for i in range(self.nb):
            X_LB = RigidTransform.Identity()
            X_LB.set_translation(config["mesh_translation"][i])
            self.X_LB_list.append(X_LB)

        # link_particle_indices_list[i] is a list of indices into self.particles.
        # The list consists of points sampled from the link indexed by
        # self.active_link_idx[i].
        self.link_particle_indices_list = [None] * len(self.active_link_idx)

        # If the infinity norm of joint torque is less than this margin, then skip
        # detecting contacts.
        self.tau_detection_margin = config["tau_detection_margin"]

        # number of positive contact detections before publishing beliefs.
        self.belief_confidence_margin = config["belief_confidence_margin"]

        self.current_contact_count = 0
        self.variance = config["standard_deviation"] ** 2

        w_cutoff = config["tau_low_pass_filter_config"]["w_cutoff"]
        if w_cutoff == "inf":
            w_cutoff = np.inf
        self.tau_filter = LowPassFilter(
            dimension=7,
            h=config["tau_low_pass_filter_config"]["h"],
            w_cutoff=w_cutoff)
        self.tau_filter.update(np.zeros(7))

    def SamplePointsOnMeshes(self):
        """
        THe probability of a point being sampled on a link is proportional to the
        surface area of the link's mesh.
        :return:
        """
        points_new = np.zeros((self.kp, 3))
        mesh_indices_new = np.zeros(self.kp, dtype=int)
        i_start = 0
        for i in range(len(self.mesh_list)):
            np_i = self.num_link_samples[i]
            i_end = i_start + np_i
            points, triangle_idx = self.mesh_list[i].sample(
                np_i, return_index=True)
            points_new[i_start:i_end] = points
            mesh_indices_new[i_start:i_end] = triangle_idx
            self.link_particle_indices_list[i] = [j for j in range(i_start, i_end)]
            i_start = i_end
        return points_new, mesh_indices_new

    def UpdateLinkPoses(self, q):
        '''
        update pose bundle (list of X_WL), link poses in world frame.
        :param q: robot joint angle.
        :return:
        '''
        self.x[:self.nq] = q
        self.context.SetDiscreteState(self.x)
        for i in range(self.nb):
            X_WL = self.plant.CalcRelativeTransform(
                self.context,
                frame_A=self.plant.world_frame(),
                frame_B=self.link_frames[i])
            self.pose_bundle[i] = X_WL

    def CalcLinkJacobian(self, q, i, point):
        """

        :param q: current robot joint angles
        :param i: link index, 0, 1, 2, ... 7
        :param point: (3,) numpy array, a point in the link's frame.
        :return:
        """
        self.x[:self.nq] = q
        self.context.SetDiscreteState(self.x)
        return self.plant.CalcFrameGeometricJacobianExpressedInWorld(
            context=self.context,
            frame_B=self.link_frames[i],
            p_BoFo_B=point)[3:]

    def RunMotionModel(self):
        """
        All calculations are done in the mesh frame (B).
        """
        points_new = np.zeros((self.kp, 3))
        indices_new = np.zeros(self.kp, dtype=int)

        for k in range(len(self.active_link_idx)):
            particles_indices = self.link_particle_indices_list[k]
            kp_k = len(particles_indices)
            if kp_k == 0:
                continue
            normals = (
                self.face_normals_list[k][
                    self.indices_in_mesh[particles_indices]])
            random_component = np.random.randn(kp_k, 3) * 0.001
            random_component += normals * 0.1
            points_new_floating = self.particles[particles_indices] + random_component
            a = -np.sign(np.sum(random_component * normals, axis=1))  # correct sign

            # ray casting
            locations, index_ray, index_tri = \
                self.mesh_list[k].ray.intersects_location(
                    ray_origins=points_new_floating,
                    ray_directions=(normals.T * a).T)

            # create points_new_projected and indices_new
            d = np.full(kp_k, np.inf)
            all_indices = np.full(kp_k, kp_k * 2)

            for i, j in enumerate(index_ray):
                l = particles_indices[j]
                d_j = np.linalg.norm(self.particles[l] - locations[i])
                if d_j < d[j]:
                    d[j] = d_j
                    points_new[l] = locations[i]
                    indices_new[l] = index_tri[i]

                all_indices[j] = j

            # if the projection of (an old point + noise) does not intersect the
            # mesh, use the old point
            for i, j in enumerate(all_indices):
                if i != j:
                    l = particles_indices[i]
                    points_new[l] = self.particles[l]
                    indices_new[l] = self.indices_in_mesh[l]
        return points_new, indices_new

    def Reset(self):
        self.has_contact = False

    def RunContactParticleFilter(self, q, tau_external_raw):
        self.tau_filter.update(tau_external_raw)
        tau_external = self.tau_filter.get_current_state()

        if np.max(np.abs(tau_external)) < self.tau_detection_margin:
            self.has_contact = False
            return

        if not self.has_contact:
            # self.particles = np.zeros((self.kp, 3))
            # self.indices = np.zeros(self.kp, dtype=int)
            points_new, indices_new = self.SamplePointsOnMeshes()
            self.has_contact = True
        else:
            # "prediction"
            points_new, indices_new = self.RunMotionModel()

        # measurement update: calculate probability for each particle.
        self.UpdateLinkPoses(q)
        optimal_values = np.full(self.kp, 0.5 * (tau_external ** 2).sum())
        f_W_particles = np.zeros_like(self.f_W_particles)

        for k, idx_link in enumerate(self.active_link_idx):
            X_WL = self.pose_bundle[idx_link]
            X_LB = self.X_LB_list[idx_link]
            for l in self.link_particle_indices_list[k]:
                # contact normal in world frame
                n_W = -X_WL.rotation().matrix().dot(
                    self.face_normals_list[k][indices_new[l]])
                vC = CalcFrictionConeRays(
                    n_W, self.friction_coefficient, self.nd)

                Jc = self.plant.CalcJacobianSpatialVelocity(
                    context=self.context,
                    with_respect_to=JacobianWrtVariable.kQDot,
                    frame_B=self.link_frames[idx_link],
                    p_BP=X_LB.multiply(points_new[l]),
                    frame_A=self.plant.world_frame(),
                    frame_E=self.plant.world_frame())[3:]

                Q_half = Jc.T.dot(vC)
                Q = Q_half.T.dot(Q_half)
                b = - Q_half.T.dot(tau_external)

                # solve QP using pre-compiled code
                emosqp.update_P(GetUpperTriangularDataAlongColumn(Q), None, 0)
                emosqp.update_lin_cost(b)
                results = emosqp.solve()

                if results[2] == 1:
                    f = results[0]
                    f_W_particles[l] = np.sum(vC * f, axis=1)
                    optimal_values[l] += (0.5 * f.dot(Q.dot(f)) + f.dot(b))
                else:
                    optimal_values[l] = np.inf

        # calc particle probabilities
        # unnormalized log probability
        log_p = -2 * optimal_values / self.variance
        log_p_max = np.max(log_p)
        if log_p_max < -5:
            # maximum likelihood is too small to be confident.
            self.Reset()
            return

        log_p -= log_p_max
        p = np.exp(log_p)
        p /= p.sum()
        p_sum = np.zeros(self.kp)
        for i in range(self.kp):
            if i == 0:
                p_sum[i] = p[0]
            else:
                p_sum[i] = p[i] + p_sum[i - 1]

        particle_link_idx = np.zeros(self.kp, dtype=int)
        for i, link_particle_idx in enumerate(self.link_particle_indices_list):
            for l in link_particle_idx:
                particle_link_idx[l] = i
            del link_particle_idx[:]

        # importance resample
        particle_link_idx_new = np.zeros(self.kp, dtype=int)
        for i in range(self.kp):
            j = np.argmax(p_sum > np.random.rand())
            self.particles[i] = points_new[j]
            self.particles_optimal_values[i] = optimal_values[j]
            self.f_W_particles[i] = f_W_particles[j]
            self.indices_in_mesh[i] = indices_new[j]
            particle_link_idx_new[i] = particle_link_idx[j]

        for i, k in enumerate(particle_link_idx_new):
            self.link_particle_indices_list[k].append(i)

    def GenerateNullBelief(self):
        cloest_points = np.array([[np.nan, np.nan, np.nan]])
        link_idx = np.array([np.nan])
        f_Ws = np.array([[np.nan, np.nan, np.nan]])
        normal = np.array([[np.nan, np.nan, np.nan]])
        triangle_id = np.array([np.nan])

        return 0, cloest_points, f_Ws, link_idx, normal, triangle_id

    def CalcBelief(self):
        if not self.has_contact:
            self.current_contact_count = 0
            return self.GenerateNullBelief()
        elif self.current_contact_count < self.belief_confidence_margin:
            self.current_contact_count += 1
            return self.GenerateNullBelief()

        std = np.std(self.particles, axis=0)
        # print(std)
        if not np.mean(std) < 0.01:
            return self.GenerateNullBelief()

        n = len(self.active_link_idx)
        # find the link with the most number of particles
        num_particles_per_link = np.zeros(n)
        for i in range(n):
            num_particles_per_link[i] = len(self.link_particle_indices_list[i])

        i_max = np.argmax(num_particles_per_link)

        mean_point = np.mean(
            self.particles[self.link_particle_indices_list[i_max]], axis=0)
        mean_point.resize((1, 3))
        closest_points, distance, triangle_id = \
            trimesh.proximity.closest_point(self.mesh_list[i_max], mean_point)

        # transform closest point from mesh frame (B) to link frame (L).
        for point in closest_points:
            point[:] = self.X_LB_list[self.active_link_idx[i_max]].multiply(point)

        link_idx = [self.active_link_idx[i_max]]
        num_contacts = 1
        normal = np.zeros((num_contacts, 3))
        normal[0] = self.face_normals_list[i_max][triangle_id[0]]
        f_Ws = np.zeros((num_contacts, 3))
        f_Ws[0] = np.mean(
            self.f_W_particles[self.link_particle_indices_list[i_max]], axis=0)
        return num_contacts, closest_points, f_Ws, link_idx, normal, triangle_id


