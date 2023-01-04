import os
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
import pybullet as p
import time
import numpy as np
import pybullet_data
import os
import pkgutil
from IPython.display import Image, Video
import cv2
from tqdm import tqdm
import pkgutil
import configparser


class PhysicalAnimation():
    def __init__(self, dirname):
        p.connect(p.DIRECT)
        egl = pkgutil.get_loader('eglRenderer')
        if (egl):
            eglPluginId = p.loadPlugin(
                egl.get_filename(), "_eglRendererPlugin")
        else:
            eglPluginId = p.loadPlugin("eglRendererPlugin")
        self.dirname = dirname
        mesh_filename = dirname + '/mesh.obj'
        self.video_filename = dirname + '/animation.avi'
        config = configparser.RawConfigParser()
        config.read(dirname + '/setting.cfg')
        # caculate the gravity center of the object
        vertices = np.loadtxt(dirname + '/vertices.txt')
        self.vertices = vertices
        tets = np.loadtxt(dirname + '/tets.txt')
        tets = tets.astype(int)
        tet_volumes = np.zeros(tets.shape[0])
        tet_centers = np.zeros((tets.shape[0], 3))
        for i in range(tets.shape[0]):
            tet_volumes[i] = np.abs(np.linalg.det(
                np.hstack((vertices[tets[i, :], :], np.ones((4, 1)))))) / 6
            tet_centers[i, :] = np.sum(
                vertices[tets[i, :], :], axis=0) / 4
        total_volume = np.sum(tet_volumes)
        gravity_center = np.sum(
            tet_centers * tet_volumes[:, None], axis=0) / total_volume
        # print('gravity center: ', gravity_center)
        # OBJ
        obj_config = dict(config.items('OBJ'))
        # print(obj_config)
        init_pos = np.array(np.matrix(obj_config['init_pos'])).ravel()
        init_ori = np.array(np.matrix(obj_config['init_ori'])).ravel()
        init_vel = np.array(np.matrix(obj_config['init_vel'])).ravel()
        # SIMULATION
        sim_config = dict(config.items('SIMULATION'))
        self.time_step = float(sim_config['time_step'])
        self.time_length = float(sim_config['time_length'])
        # RENDERING
        render_config = dict(config.items('RENDERING'))
        # print(render_config)
        fps = float(render_config['fps'])
        self.animation_rate = 1.0 / fps
        camTargetPos = np.array(
            np.matrix(render_config['camtargetpos'])).ravel()
        yaw = float(render_config['yaw'])
        pitch = float(render_config['pitch'])
        roll = float(render_config['roll'])
        camDistance = float(render_config['camdistance'])
        upAxisIndex = int(render_config['upaxisindex'])
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch,
                                                              roll, upAxisIndex)
        self.pixelWidth = int(render_config['pixelwidth'])
        self.pixelHeight = int(render_config['pixelheight'])
        nearPlane = float(render_config['nearplane'])
        farPlane = float(render_config['farplane'])
        fov = float(render_config['fov'])
        aspect = self.pixelWidth / self.pixelHeight
        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov, aspect, nearPlane, farPlane)

        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane100.urdf", useMaximalCoordinates=True)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                  fileName=mesh_filename,
                                                  collisionFramePosition=[
                                                      0, 0, 0],
                                                  meshScale=[1, 1, 1])
        obj_id = p.createMultiBody(baseMass=1,
                                   baseInertialFramePosition=gravity_center,
                                   baseCollisionShapeIndex=collisionShapeId,
                                   basePosition=init_pos,
                                   baseOrientation=init_ori,
                                   useMaximalCoordinates=True)
        p.setGravity(0, 0, -9.8)
        p.resetBaseVelocity(obj_id, linearVelocity=init_vel)
        p.changeDynamics(obj_id, -1, lateralFriction=0.01,
                         rollingFriction=0.01, restitution=0.01)
        p.changeDynamics(plane_id, -1, lateralFriction=0.01,
                         rollingFriction=0.01, restitution=0.01)

        p.setTimeStep(self.time_step)
        self.obj_id = obj_id
        self.plane_id = plane_id
        time_step_num = int(self.time_length / self.time_step)
        self.time_step_num = time_step_num

    def generate(self):
        print('Generating animation...')
        motion_data = np.zeros((self.time_step_num, 7))
        contact_data = np.zeros((self.time_step_num, 4, 7))

        fps = int(1 / self.animation_rate)
        animation_step = 0
        video = cv2.VideoWriter(self.video_filename, cv2.VideoWriter_fourcc(
            *'DIVX'), fps, (self.pixelWidth, self.pixelHeight))
        p.stepSimulation()
        for i in tqdm(range(self.time_step_num)):
            p.stepSimulation()
            Pos, Ori = p.getBasePositionAndOrientation(self.obj_id)
            motion_data[i, :3] = Pos
            motion_data[i, 3:] = Ori
            points_data = p.getContactPoints(
                bodyA=self.plane_id, bodyB=self.obj_id)
            idx = 0
            for data in points_data:
                contact_data[i, idx, :3] = data[6]  # positionOnB
                contact_data[i, idx, 3:6] = data[7]  # contactNormalOnB
                contact_data[i, idx, 6] = data[9]  # appliedImpulse
                # print(data[6])
                idx += 1
            if (animation_step * self.animation_rate < self.time_step * i):
                animation_step += 1
                img_arr = p.getCameraImage(
                    self.pixelWidth, self.pixelHeight, self.viewMatrix, self.projectionMatrix)
                w = img_arr[0]  # width of the image, in pixels
                h = img_arr[1]  # height of the image, in pixels
                rgb = img_arr[2]  # color data RGB
                dep = img_arr[3]  # depth data
                np_img_arr = np.reshape(rgb, (h, w, 4)).astype(np.uint8)
                frame = np_img_arr[:, :, :3]
                video.write(frame)
        video.release()
        mp4_file = self.video_filename.replace('.avi', '.mp4')
        os.popen(
            f'ffmpeg -hide_banner -loglevel error -i {self.video_filename} {mp4_file} -y; rm {self.video_filename}')
        self.motion = motion_data
        self.contact = contact_data

    # convert the global contact point to the local contact point
    # obj_pos: the position of the object (x, y, z)
    # obj_ori: the orientation of the object (x, y, z, w)
    def global_to_local(self, world_point, obj_pos, obj_ori):
        local_point = world_point - obj_pos
        # rotate the local point to the world frame
        local_point = np.array(p.getMatrixFromQuaternion(obj_ori)).reshape(
            (3, 3)).T.dot(local_point)
        return local_point

    def local_to_global(self, local_point, obj_pos, obj_ori):
        world_point = np.array(p.getMatrixFromQuaternion(obj_ori)).reshape(
            (3, 3)).dot(local_point)
        world_point += obj_pos
        return world_point

    # find closest vertex to the contact point
    def post_process_contacts(self):
        print('post processing contact forces....')
        from scipy.spatial import KDTree
        kdtree = KDTree(self.vertices)
        y_range = self.vertices[:, 1].max(), self.vertices[:, 1].min()
        # time, x, y, z, qx, qy, qz, qw
        motion_output = np.zeros((self.time_step_num, 8))
        # time, vertex_id, nx, ny, nz, amount
        contact_output = np.zeros((self.time_step_num * 4, 6))
        for i in tqdm(range(self.time_step_num)):
            for j in range(4):
                if (self.contact[i, j, 6] != 0):
                    contact_point = self.global_to_local(
                        self.contact[i, j, :3], self.motion[i, :3], self.motion[i, 3:])
                    contact_normal = self.global_to_local(
                        self.contact[i, j, 3:6], [0, 0, 0], self.motion[i, 3:])
                    constact_vertex_id = kdtree.query(contact_point)[1]
                    contact_output[i * 4 + j, 0] = i * self.time_step
                    contact_output[i * 4 + j, 1] = constact_vertex_id
                    contact_output[i * 4 + j, 2:5] = contact_normal
                    contact_output[i * 4 + j, 5] = self.contact[i, j, 6]

            motion_output[i, 0] = i * self.time_step
            motion_output[i, 1:8] = self.motion[i, :]
        # remove the zero force contact points
        contact_output = contact_output[contact_output[:, 5] != 0]
        self.motion = motion_output
        self.contact = contact_output
        np.savetxt(self.dirname + '/motion.txt', motion_output)
        np.savetxt(self.dirname + '/contact.txt', contact_output)
