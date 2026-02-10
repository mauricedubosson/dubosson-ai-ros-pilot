import numpy as np
import torch
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import subprocess
import time
import psutil  # For CPU monitoring

# Parameters
N = 256  # Reduced for CPU efficiency (original 1024)
generations = 1000
cp_bias = 0.0245
scalar_energy = 0.0075
mem_gain = 0.8
senescence_threshold = 0.8
protein_boost = 0.2
device = torch.device("cpu")
obstacle_threshold = 1.0  # Distance (m) for wall shock
shock_penalty = -20.0  # Negative shock to V_t if too close to wall

# ... (Rest of the code as before - ScalarField, vectorized_dynamics, NeuralNet)

class NeuralNet(nn.Module):  # Updated with tanh normalization
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))  # Normalized [-1,1] for stable angular.z

net = NeuralNet().to(device)

# Gazebo Launch
def launch_gazebo():
    subprocess.Popen(["roslaunch", "turtlebot3_gazebo", "turtlebot3_world.launch"])
    time.sleep(5)

launch_gazebo()

# ROS Node
rospy.init_node('alife_robot_controller')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
rate = rospy.Rate(10)

# Lidar Callback for Closing the Loop (Shock if Wall Close)
def lidar_callback(data):
    min_distance = min(data.ranges)
    if min_distance < obstacle_threshold:
        affected = np.random.choice(N, int(0.5 * N))  # Affect 50% randomly
        V_t[affected] += shock_penalty  # Negative shock to fitness

sub = rospy.Subscriber('/scan', LaserScan, lidar_callback)

# Evolution Loop with ROS and CPU Management
fitness_history = []
population_history = [N]

for gen in range(1, generations + 1):
    # CPU Check (Pause if Overloaded)
    if psutil.cpu_percent() > 90:
        time.sleep(1)  # Pause 1s to cool CPU

    # ... (Rest of the loop as before - Perturbations, Dynamics, Fitness, Senescence, Mutations, Cooperation, Neural Decision, Adaptability, Replication, Scalar Shock)

    # ROS Control
    twist = Twist()
    twist.linear.x = fitness.mean().item()
    twist.angular.z = np.mean(decisions) * 0.5
    pub.publish(twist)
    rate.sleep()

    fitness_history.append(fitness.mean().item())
    population_history.append(N)

    if gen % 100000 == 0:
        print(f"Generation {gen}: Population {N}, Avg Fitness {fitness.mean().item()}")

# Save and Plot (as before)
