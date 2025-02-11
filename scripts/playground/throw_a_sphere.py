# Ref: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/run_rigid_object.html
# Ref: source/extensions/omni.isaac.lab/test/sensors/check_imu_sensor.py
# Ref: source/extensions/omni.isaac.lab/test/assets/test_rigid_object.py

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher
# launch omniverse app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
# Then you’ll probably have to start the SimulationApp() before importing those packages
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import RigidObjectCfg, RigidObject
import isaaclab.utils.math as math_utils
import omni

import torch
"""Rest everything follows."""
from isaaclab.sim import SimulationContext

from omni.physx import get_physx_interface
from pxr import UsdPhysics, Gf

def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    origins = [[-1.0, 0.0, 0.05]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Rigid Object
    sphere_cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    balls = RigidObject(cfg=sphere_cfg)

    # return the scene information
    return balls
   
def run_simulator(sim: sim_utils.SimulationContext, entities: RigidObject):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    
    """Reset firstly and only once. It must always be called the first time before stepping the simulator. Otherwise, the simulation 
    handles are not initialized properly."""
    sim.reset()
    
    # you’ll probably have to sim reset() before rigid object operations
    sphere_object = entities
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        print(f"Simulating count: {count}")
        # apply a force to throw the sphere
        if count == 100:
            # Find bodies to apply the force
            body_ids, body_names = sphere_object.find_bodies(".*")
            # Sample a force equal to the weight of the object
            external_wrench_b = torch.zeros(sphere_object.num_instances, len(body_ids), 6, device=sim.device)
            # Every 2nd cube should have a force applied to it
            external_wrench_b[0::2, :, 0] = 9.81 * sphere_object.root_physx_view.get_masses()[0]*25
            external_wrench_b[0::2, :, 2] = 9.81 * sphere_object.root_physx_view.get_masses()[0]*25
            sphere_object.set_external_force_and_torque(
                forces=external_wrench_b[..., :3], 
                torques=external_wrench_b[..., 3:], body_ids=body_ids)

            # apply sim data
            sphere_object.write_data_to_sim()

        # perform step
        sim.step()

        # update sim-time
        sim_time += sim_dt
        count += 1

        # update buffers
        sphere_object.update(sim_dt)

        # print the root position
        if count % 50 == 0:
            print(f"Root position (in world): {sphere_object.data.root_link_state_w[:, :3]}")


if __name__ == "__main__":
   # get simulation context
   simulation_context = SimulationContext()
   # reset and play simulation
#    simulation_context.reset()
   # Now we are ready!
   print("[INFO]: Setup complete...")
    
   balls = design_scene()
   
   run_simulator(simulation_context, balls)
#    # step simulation
#    simulation_context.step()
   
#    # stop simulation
#    simulation_context.stop()

   # close the simulation
   simulation_app.close()