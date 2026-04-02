import time
import sys

# --- Drake Imports ---
from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser
)

# --- MuJoCo Imports ---
import mujoco
import mujoco.viewer

# Define the paths (You can change this to g1_with_hands.xml if the scene fails in Drake)
XML_PATH = "/home/sid/projects25/src/bheema/unitree_g1/scene_with_hands.xml" 

def test_drake_parsing(xml_path):
    print("\n--- 1. TESTING DRAKE PARSER ---")
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    
    try:
        # Drake natively supports URDF, SDF, and MJCF (MuJoCo XML)
        parser.AddModels(xml_path)
        plant.Finalize()
        
        # If it reaches here, Drake successfully accepted the math and geometries!
        print("✅ SUCCESS: Drake successfully parsed and finalized the model!")
        print(f"  -> Total Bodies extracted: {plant.num_bodies()}")
        print(f"  -> Total Joints extracted: {plant.num_joints()}")
        print(f"  -> Total Actuators extracted: {plant.num_actuators()}")
        
        # Calculate total mass to verify it matches your expectations (~35kg)
        context = plant.CreateDefaultContext()
        total_mass = plant.CalcTotalMass(context)
        print(f"  -> Total Mass calculated: {total_mass:.2f} kg\n")
        
    except Exception as e:
        print("❌ FAILED: Drake threw an error while parsing the XML.")
        print("Drake is much stricter than MuJoCo regarding inertias and collision geometries.")
        print(f"Error Details:\n{e}\n")
        sys.exit(1)

def launch_mujoco_viewer(xml_path):
    print("--- 2. LAUNCHING MUJOCO VIEWER ---")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print("✅ SUCCESS: MuJoCo parsed the model.")
        
        print("Launching interactive viewer... (Press ESC or close window to exit)")
        # Launch the standard passive viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)
                
    except Exception as e:
        print("❌ FAILED: MuJoCo threw an error while parsing.")
        print(f"Error Details:\n{e}")

if __name__ == "__main__":
    # Run the Drake validation first
    test_drake_parsing(XML_PATH)
    
    # If Drake succeeds, launch the MuJoCo viewer to visualize it
    launch_mujoco_viewer(XML_PATH)