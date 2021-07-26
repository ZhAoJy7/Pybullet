import pybullet as p
import time
import math
import numpy as np
from datetime import datetime

# from trajectory import *


# pybullet 3D simulator parameters initialization
clid = p.connect(p.SHARED_MEMORY)
if (clid < 0): p.connect(p.GUI)
p.loadURDF("plane.urdf", [0, 0, -0.3])
kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
kukaEndEffectorIndex = 6
numJoints = p.getNumJoints(kukaId)
if (numJoints != 7): exit()
useSimulation = 1  # use dynamic motor control (can interact with robot in simulator)
useRealTimeSimulation = 1
p.setRealTimeSimulation(useRealTimeSimulation)
p.setGravity(0, 0, 0)
trailDuration = 5  # trailDuration is duration (in seconds) after debug lines will be removed automatically

# Inverse Kinematics control's parameters
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]  # lower limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]  # upper limits for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]  # joint ranges for null space
rp = [0, 0, 0, 0, 0, 0, 0]  # rest poses for null space
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # joint damping coefficients
ikSolver = 0

# Initialize robot joint states
for i in range(numJoints):
    p.resetJointState(kukaId, i, rp[i])


# Define simple IK_write
def S_IK_write(letter='0', plan='vertical', orientation='front', useNullSpace=True, resample=0.7):
    traj = -np.load('python_data/2Dletters/' + letter + '.npy')
    # Add some frames at the start position of trajetory, in order for robot to pause a bit before writing new letter
    # 把traj[0]多复制几个加在轨迹最前面,让他在写字母前先停留一会
    traj = np.concatenate((np.tile(traj[0], (int(resample * 2000), 1)), np.array(traj)))

    # Resample trajetory to make robot writing slower
    # 线性插值,多一些点,减慢写字速度
    traj_1 = - np.interp(np.arange(0, len(traj), resample), np.arange(0, len(traj)), traj[:, 0])
    traj_2 = np.interp(np.arange(0, len(traj), resample), np.arange(0, len(traj)), traj[:, 1])

    # Rescale trajectory to adapt to robot working area
    traj_1 = np.interp(traj_1, (np.min(traj_1), np.max(traj_1)), (-0.15, 0.15))
    traj_2 = np.interp(traj_2, (np.min(traj_2), np.max(traj_2)), (0.3, 0.6))

    if orientation == 'front':
        orn = p.getQuaternionFromEuler([0, -0.5 * math.pi, 0])  # Robot's end effector point orientation (point down)
    elif orientation == 'down':
        orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # Robot's end effector point orientation (point down)
    else:
        raise ('Robot end effector orientation parameter error')

    prevPose = [0, 0, 0]
    prevPose1 = [0, 0, 0]
    # Begin robot writing letter
    for n in range(len(traj_1)):
        # Convert 2D letter trajecotry to 3D trajecotry in vertical plan or horizonal plan
        if plan == 'vertical':
            pos = [-0.6, traj_1[n], traj_2[n]]
        elif plan == 'horizontal':
            pos = [traj_1[n] - 0.3, traj_2[n], 0.4]
        else:
            raise ('Robot trajectory plan parameter error')

        if useNullSpace == True:
            # Calculate robot's joint poses using null-space inverse kinematics control
            jointPoses = p.calculateInverseKinematics(kukaId,
                                                      kukaEndEffectorIndex,
                                                      pos,
                                                      orn,
                                                      ll,
                                                      ul,
                                                      jr,
                                                      rp)
        else:
            # Calculate robot's joint poses using inverse kinematics without null-space control (use joint damping instead)
            jointPoses = p.calculateInverseKinematics(kukaId,
                                                      kukaEndEffectorIndex,
                                                      pos,
                                                      orn,
                                                      jointDamping=jd,
                                                      solver=ikSolver,
                                                      maxNumIterations=100,
                                                      residualThreshold=.01)

        # Set dynamic motion control for each joint
        if (useSimulation):
            for i in range(numJoints):
                p.setJointMotorControl2(bodyIndex=kukaId,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=jointPoses[i],
                                        targetVelocity=0,
                                        force=500,
                                        positionGain=0.03,
                                        velocityGain=1)
        # No use dynamic motion control
        else:
            # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
            for i in range(numJoints):
                p.resetJointState(kukaId, i, jointPoses[i])

        # Get link state of robot arm
        ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
        # Draw target letter trajectory and robot trajectory inside simulator
        if n > 2000:
            p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 2, trailDuration)  # Draw red line of robot trajectory
            p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 5,  trailDuration)  # Draw black line of target letter trajectory
        prevPose = pos
        prevPose1 = ls[4]


nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
for i in nums:
    S_IK_write(letter=i, plan='vertical', orientation='front', useNullSpace=True, resample=0.7)
    time.sleep(5)

p.disconnect()
