# dubosson-ai-ros-pilot
Dubossonien ALife Pilot adapted to robotics with ROS and Gazebo for resilient autonomous AI
OverviewOpen-source Python pilot applying Dubosson's theory to robotics: thought as Kantian selection in "membranes" for adaptive, eternal resilience. Integrates ROS and Gazebo for real/simulated robots (e.g., TurtleBot), simulating protocell intelligence to control movement and decisions.FeaturesALife protocells guiding robot commands (speed/turn based on fitness/decisions).
Cosmic grafting (CP bias, scalar field) for autonomy.
Emergent traits: Senescence, neural cognition, adaptability.
ROS integration: Publishes to /cmd_vel for real-time control.
Gazebo simulation: Launches TurtleBot world for testing.

InstallationInstall ROS (Noetic/Humble) and Gazebo.
pip install numpy torch rospy.
Install TurtleBot packages: sudo apt install ros-noetic-turtlebot3-gazebo (adapt for your ROS version).
Launch ROS: roscore.

UsageRun python dubosson_ai_ros_pilot.py – launches Gazebo (TurtleBot world) and controls robot via ALife.
Adjust generations for simulation length.
Outputs: NPY files for histories, PNG plot for evolution.

Code Structuredubosson_ai_ros_pilot.py: Main loop with ALife, ROS publisher, and Gazebo launch.

Results (1M Generations Example)Swarm Size: Stabilizes ~25k-35k.
Fitness: ~0.26-0.30.
Emergent: Quantum senescence, eternal robot adaptation.

ContributingFork the repo, experiment (e.g., with real robots), PR welcome. DM @mauricedubosson
 on X.LicenseCreative Commons Attribution-NonCommercial-ShareAlike 4.0 International LicenseCopyright (c) 2026 Maurice DubossonThis work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.You are free to:Share — copy and redistribute the material in any medium or format
Adapt — remix, transform, and build upon the material

Under the following terms:Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
NonCommercial — You may not use the material for commercial purposes.
ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.For commercial use or other inquiries, contact Maurice Dubosson: @mauricedubosson on X.

