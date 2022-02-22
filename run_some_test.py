
#%%
import trimesh
import meshcat
import lcm
from contact_particle_filter.contact_particle_filter_core import (
    ContactParticleFilter)
from contact_particle_filter.lcmt_contact_particle_filter import lcmt_contact_position
from drake import lcmt_iiwa_status

from contact_particle_filter.utils import *
from contact_particle_filter.contact_visualizer import *

q0 = [0, 0, 0, -1.75, 0, 1.0, 0]

cpf = ContactParticleFilter()
cpf.UpdateLinkPoses(q0)

# override the true contact point with a point on link 7
link_idx = 7
mesh = trimesh.load(
    "./iiwa7_shifted_meshes/link_7_w_ball.obj")
points_true = np.array([[0, 0, 0.11]])
points_true, distance, triangle_id = \
    trimesh.proximity.closest_point(mesh, points_true)
J_WL7 = cpf.plant.CalcFrameGeometricJacobianExpressedInWorld(
    context=cpf.context,
    frame_B=cpf.link_frames[link_idx],
    p_BoFo_B=points_true[0])[3:]

X_WL7 = cpf.pose_bundle[link_idx]
X_WB7 = X_WL7.multiply(cpf.X_LB_list[link_idx])

normal_true_world = X_WB7.rotation().multiply(
    mesh.face_normals[triangle_id[0]])
point_true_world = X_WB7.multiply(points_true[0])
f_W_true = -10 * normal_true_world
tau_external = J_WL7.T.dot(f_W_true)
#%%
vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
DrawRobot(vis, cpf.pose_bundle, cpf.X_LB_list)
DrawArrow(vis, "arrow_true",
          origin=point_true_world,
          direction=normal_true_world,
          color=0x0000ff, length=0.2)

#%%
cpf.Reset()
while True:
    cpf.RunContactParticleFilter(q0, tau_external)
    result = cpf.CalcBelief()
    if result[0] > 0:
        break

print(np.mean(cpf.particles, axis=0))
print(np.var(cpf.particles, axis=0))

num_contacts, closest_points, f_Ws, link_idx, normal, triangle_id = result
f_W_detected = f_Ws[0]

DrawPointCloud(vis,
               name="link6_points",
               points=cpf.particles,
               color=[1, 0, 0],
               X_WB=cpf.pose_bundle[link_idx[0]].multiply(
             cpf.X_LB_list[link_idx[0]]).matrix())

DrawArrow(vis, "arrow_belief",
          origin=cpf.pose_bundle[link_idx[0]].multiply(closest_points[0]),
          direction=-f_W_detected,
          color=0xff0000, length=0.2)

#%% compare detected and true contact forces.
f_W_true_norm = np.linalg.norm(f_W_true)
f_W_detected_norm = np.linalg.norm(f_W_detected)
u_W_true = f_W_true / f_W_true_norm
u_W_detected = f_W_detected / f_W_detected_norm
print("frue force norm: ", f_W_true_norm)
print("detected force norm: ", f_W_detected_norm)
print("inner product: ", u_W_true.dot(u_W_detected))


#%%
import cProfile
cProfile.runctx(
    "cpf.RunContactParticleFilter(q0, tau_external)",
    globals(), locals(), filename="stats")