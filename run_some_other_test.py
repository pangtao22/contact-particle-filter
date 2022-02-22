
import pickle
import trimesh
import meshcat
import lcm
from contact_particle_filter.contact_particle_filter_core import (
    ContactParticleFilter)
from contact_particle_filter.lcmt_contact_particle_filter import lcmt_contact_position
from drake import lcmt_iiwa_status

from pydrake.multibody.tree import JacobianWrtVariable
from contact_particle_filter.contact_visualizer import *

#%%
test_result_file_name = "../cpf_mc_test Tue Apr 28 04:59:42 2020"
test_result = pickle.load(open(test_result_file_name, "rb"))

test_idx = 6
q0 = test_result["joint_angles"][test_idx]
link_idx = test_result["true_contact_link"][test_idx]
p_LC_L_true = test_result["true_contact_point_p_LC_L"][test_idx]
f_W_true = test_result["true_contact_force_f_W"][test_idx]
tau_external = test_result["tau_ext"][test_idx]

cpf = ContactParticleFilter()
cpf.UpdateLinkPoses(q0)

X_WL7 = cpf.pose_bundle[link_idx]
X_WB7 = X_WL7.multiply(cpf.X_LB_list[link_idx])
point_true_world = X_WB7.multiply(p_LC_L_true)

vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
DrawRobot(vis, cpf.pose_bundle, cpf.X_LB_list)
DrawArrow(vis, "arrow_true",
          origin=point_true_world,
          direction=-f_W_true,
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