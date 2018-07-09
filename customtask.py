import numpy as np
from physics_sim import PhysicsSim

class CustomTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        
        # For tracking differences in z (for hover)
        self.reward_last_z = target_pos[2] if target_pos is not None else 10.
        self.reward_this_z = target_pos[2] if target_pos is not None else 10.

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""

        # Premise is sound, as we want to reward highest when sim.pose x,y,z is 
        # essentially equal target_pos x,y,z (making the product of discount rate
        # and pose diff essentially 0 -- therefore, reward would be close to 1).
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos).sum())
        
        # rrm - discounting the error
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos).sum())
        reward = 2.-.2*(abs(self.sim.pose[:3] - self.target_pos).sum())
        
        # By experience in running, this reward gets negative quickly. We need to
        # scale it, so it can hopefully learn more efficiently.
        # Let's see what happens when we just cap the negative reward at -1
        """
        if reward > 1.0:
            print("Reward is > 1: {0}".format(reward))
            reward = 1.0
        elif reward < -1.0:
            print("Reward is < 1: {0}".format(reward))
            reward = -1.0
        """

        # Works pretty well... Trying something different below
        """
        if reward > 0 and reward < 0.5:
            reward = reward * 2
        elif reward > 0.5:
            reward = reward * 4
        elif reward < -1.0:
            #print("Reward is < 1: {0}".format(reward))
            reward = -1.0
        """

        # Works well, but what if we provide extra reward (or penalize more) based on z coordinate (for hovering)
        """
        absoluteZDiff = abs(self.sim.pose[2] - self.target_pos[2])
        if reward > 0 and reward < 0.5 and absoluteZDiff < 1:
            reward = reward * 3
        elif reward >= 0.5 and reward < 0.8 and absoluteZDiff < 1:
            reward = reward * 4
        elif reward >= 0.8 and absoluteZDiff < 1:
            reward = reward * 5
        elif reward > -1.0 and absoluteZDiff > 2:
            reward = -3.0 # penalize more for bad z
        else:
            reward = -1.0 # Cap it here
        """
            
        # Instead of comparing to target z, compare to last z
        origTargetZDiff = abs(self.reward_last_z - self.target_pos[2])
        self.reward_last_z = self.reward_this_z
        self.reward_this_z = self.sim.pose[2]
        
        # diff between current z and last z
        lastZDiff = abs(self.reward_last_z - self.reward_this_z)
        # diff betwen current z and target z
        targetZDiff = abs(self.reward_this_z - self.target_pos[2])
        
        """
        if lastZDiff < 0.1:
            if reward > 0 and reward < 0.5:
                reward = 0.5
            elif reward >= 0.5 and reward < 0.8:
                reward = 0.8
            elif reward >= 0.8 and reward < 1:
                reward = 1.0
        elif reward < -1.0:
            reward = -1.0 # Cap it here

        if reward > 0 and targetZDiff < 2:
            reward = reward * 1.2

        if (targetZDiff < origTargetZDiff):
            if reward > 0:
                reward = reward * 1.5
            else:
                reward = reward * 0.5
        """
        
        if reward < -1.0:
            reward = -1.0
            
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    
    def getSimXYZ(self):
        return self.sim.pose
    