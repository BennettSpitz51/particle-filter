#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header
import math
import tf.transformations as tf

# Global variables to store the current odometry data and belief state
previous_odom = None
belief_state = (0.0, 0.0, 0.0)
particle_filter = None
last_scan_received = False  # Flag to ensure odom only processes after a scan update

class ParticleFilter:
    def __init__(self, num_particles=500):
        # Load the precomputed likelihood field (distance matrix)
        self.likelihood_field = np.load('likelihood_field.npy')
        self.z_max = 15.0
        self.sigma_hit = 0.2
        self.z_hit = 0.8
        self.z_random = 0.2
        self.num_particles = num_particles
        self.particles = self.initialize_particles()
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.particles_pub = rospy.Publisher('/particle_filter/particles', PoseArray, queue_size=10)

    def initialize_particles(self):
        # Initialize particles around the robot's initial belief state
        x_init, y_init, theta_init = belief_state
        return np.array([[np.random.normal(x_init, 1.0),  # 1.0 is a standard deviation; adjust as needed
                          np.random.normal(y_init, 1.0), 
                          np.random.uniform(-np.pi, np.pi)]
                         for _ in range(self.num_particles)])

    def motion_model(self, odom_data, particle):
        delta_rot1, delta_trans, delta_rot2 = odom_data
        x, y, theta = particle
        alpha1, alpha2, alpha3, alpha4 = 0.02, 0.02, 0.01, 0.01

        # Add noise to each motion component to simulate the uncertainty in the robot's movements
        delta_rot1_hat = delta_rot1 + np.random.normal(0, alpha1 * abs(delta_rot1) + alpha2 * abs(delta_trans))
        delta_trans_hat = delta_trans + np.random.normal(0, alpha3 * abs(delta_trans) + alpha4 * (abs(delta_rot1) + abs(delta_rot2)))
        delta_rot2_hat = delta_rot2 + np.random.normal(0, alpha1 * abs(delta_rot2) + alpha2 * abs(delta_trans))

        # Update particle position with noisy motion model using particle's own orientation
        x_prime = x + delta_trans_hat * np.cos(theta + delta_rot1_hat)
        y_prime = y + delta_trans_hat * np.sin(theta + delta_rot1_hat)
        theta_prime = theta + delta_rot1_hat + delta_rot2_hat

        return np.array([x_prime, y_prime, theta_prime])

    def update_particles_with_motion(self, odom_data):
        # Update each particle's position based on the odometry reading with motion noise
        self.particles = np.array([self.motion_model(odom_data, particle) for particle in self.particles])

    def sensor_model(self, z_t, particle):
        q = 1.0
        x, y, theta = particle
        for k, z_tk in enumerate(z_t):
            if z_tk == self.z_max:
                continue
            theta_k = theta + k * np.pi / 180
            x_zk = x + z_tk * math.cos(theta_k)
            y_zk = y + z_tk * math.sin(theta_k)
            dist_sq = self.get_min_distance_squared(x_zk, y_zk)
            prob_hit = (1 / (math.sqrt(2 * np.pi * self.sigma_hit ** 2))) * \
                        math.exp(-dist_sq / (2 * self.sigma_hit ** 2))
            q *= (self.z_hit * prob_hit + self.z_random / self.z_max)
        return q

    def update_weights_with_sensor(self, z_t):
        # Update weights based on sensor readings and normalize
        self.weights = np.array([self.sensor_model(z_t, particle) for particle in self.particles])
        self.weights += 1.e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)  # Normalize weights

    def resample_particles(self):
        # Resampling with replacement based on particle weights
        indices = np.arange(self.num_particles)
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # Ensure sum is exactly 1.0
        resampled_indices = np.searchsorted(cumulative_sum, np.random.rand(self.num_particles))
        self.particles = self.particles[resampled_indices]

        # Add small noise to particles to prevent perfect tracking
        self.particles[:, 0] += np.random.normal(0, 0.1, self.num_particles)
        self.particles[:, 1] += np.random.normal(0, 0.1, self.num_particles)
        self.weights.fill(1.0 / self.num_particles)  # Reset weights

    def publish_particles(self):
        rospy.loginfo("Publishing particles...")
        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = 'map'  # Ensure this matches your RViz Fixed Frame
        
        for particle in self.particles:
            pose = Pose()
            pose.position.x = particle[0]
            pose.position.y = particle[1]
            quaternion = tf.quaternion_from_euler(0, 0, particle[2])
            pose.orientation.x = quaternion[0]
            pose.orientation.y = quaternion[1]
            pose.orientation.z = quaternion[2]
            pose.orientation.w = quaternion[3]
            pose_array.poses.append(pose)
        
        self.particles_pub.publish(pose_array)

    def get_min_distance_squared(self, x_zk, y_zk):
        grid_x = int(np.clip(x_zk, 0, self.likelihood_field.shape[1] - 1))
        grid_y = int(np.clip(y_zk, 0, self.likelihood_field.shape[0] - 1))
        return self.likelihood_field[grid_y, grid_x] ** 2

def odom_callback(odom_msg):
    global previous_odom, belief_state, particle_filter, last_scan_received

    if not last_scan_received:
        return

    rate = rospy.Rate(1)  # Adjust rate as needed
    rate.sleep()

    rospy.loginfo("Start odom")
    position = odom_msg.pose.pose.position
    orientation = odom_msg.pose.pose.orientation
    theta = euler_from_quaternion(orientation)

    if previous_odom is None:
        previous_odom = (position.x, position.y, theta)
        return

    prev_x, prev_y, prev_theta = previous_odom
    delta_trans = np.sqrt((position.x - prev_x) ** 2 + (position.y - prev_y) ** 2)
    delta_rot1 = np.arctan2(position.y - prev_y, position.x - prev_x) - prev_theta
    delta_rot2 = theta - prev_theta - delta_rot1

    previous_odom = (position.x, position.y, theta)
    current_odom = (delta_rot1, delta_trans, delta_rot2)
    
    # Update belief state and particles based on odometry
    belief_state = particle_filter.motion_model(current_odom, belief_state)
    particle_filter.update_particles_with_motion(current_odom)

    last_scan_received = False

def euler_from_quaternion(orientation):
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    theta = np.arctan2(siny_cosp, cosy_cosp)
    return theta

def scan_callback(scan_msg):
    global last_scan_received, belief_state, particle_filter
    rospy.loginfo("Start scan")
    
    ranges = np.array(scan_msg.ranges)
    particle_filter.update_weights_with_sensor(ranges)
    particle_filter.resample_particles()
    particle_filter.publish_particles()
    rospy.loginfo(f"Likelihood of current particle: {np.mean(particle_filter.weights)}")

    last_scan_received = True  # Flag that a scan was received


def main():
    global particle_filter
    
    rospy.init_node('particle_filter')
    rospy.loginfo("Particle Filter Node Initialized")
    
    particle_filter = ParticleFilter(num_particles=500)  # Reduced particles for performance

    # Subscribe to the /odom topic to receive odometry data
    rospy.Subscriber('/odom', Odometry, odom_callback, queue_size=10)
    
    # Subscribe to the /scan topic to receive LiDAR data
    rospy.Subscriber('/scan', LaserScan, scan_callback, queue_size=10)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

