# Ref: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/run_rigid_object.html

"""Launch Isaac Sim Simulator first."""
from omni.isaac.lab.app import AppLauncher
# launch omniverse app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import omni.isaac.lab.sim as sim_utils
# Then you’ll probably have to start the SimulationApp() before importing those packages
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.assets import RigidObjectCfg, RigidObject



"""Rest everything follows."""
from omni.isaac.lab.sim import SimulationContext

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
    origins = [[0.0, 0.0, 0.05]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Rigid Object
    cone_cfg = RigidObjectCfg(
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
    sphere_object = RigidObject(cfg=cone_cfg)

    # return the scene information
    scene_entities = {"sphere": sphere_object}
    return scene_entities, origins
   
if __name__ == "__main__":
   # get simulation context
   simulation_context = SimulationContext()
   # reset and play simulation
   simulation_context.reset()
   # Now we are ready!
   print("[INFO]: Setup complete...")
    
   design_scene()
   
   # step simulation
   simulation_context.step()
   
   # stop simulation
   simulation_context.stop()

   # close the simulation
   simulation_app.close()