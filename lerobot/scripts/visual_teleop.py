# -*- coding: utf-8 -*-
# @Time    : 2024/10/14
# @Author  : wenshao
# @Project : WiLoR-mini
# @FileName: test_wilor_pipeline.py

"""
you need to install trimesh and pyrender if you want to render mesh
pip install trimesh
pip install pyrender
"""

import os
import pdb
import time

import trimesh
import pyrender
import numpy as np
import torch


def create_raymond_lights():
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes


def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses


def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)


def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))


def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


class Renderer:

    def __init__(self, faces: np.array):
        """
        Wrapper around the pyrender renderer to render MANO meshes.
        Args:
            cfg (CfgNode): Model config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """

        # add faces that make the hand mesh watertight
        faces_new = np.array([[92, 38, 234],
                              [234, 38, 239],
                              [38, 122, 239],
                              [239, 122, 279],
                              [122, 118, 279],
                              [279, 118, 215],
                              [118, 117, 215],
                              [215, 117, 214],
                              [117, 119, 214],
                              [214, 119, 121],
                              [119, 120, 121],
                              [121, 120, 78],
                              [120, 108, 78],
                              [78, 108, 79]])
        faces = np.concatenate([faces, faces_new], axis=0)
        self.faces = faces
        self.faces_left = self.faces[:, [0, 2, 1]]

    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9),
                            rot_axis=[1, 0, 0], rot_angle=0, is_right=1):
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces_left.copy(),
                                   vertex_colors=vertex_colors)
        # mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())

        rot = trimesh.transformations.rotation_matrix(
            np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render_rgba(
            self,
            vertices: np.array,
            cam_t=None,
            rot=None,
            rot_axis=[1, 0, 0],
            rot_angle=0,
            camera_z=3,
            # camera_translation: np.array,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0, 0, 0),
            render_res=[256, 256],
            focal_length=None,
            is_right=None,
    ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length / render_res[1]])
        if is_right:
            mesh_base_color = mesh_base_color[::-1]
        mesh = self.vertices_to_trimesh(vertices, np.array([0, 0, 0]), mesh_base_color, rot_axis, rot_angle,
                                        is_right=is_right)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            # node = pyrender.Node(
            #     name=f"light-{i:02d}",
            #     light=pyrender.DirectionalLight(color=color, intensity=intensity),
            #     matrix=matrix,
            # )
            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)


def test_wilor_image_pipeline():
    import cv2
    import torch
    import numpy as np
    import os
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

    LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
    img_path = "lerobot/scripts/img8.png"
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
    for _ in range(20):
        t0 = time.time()
        outputs = pipe.predict(image)
        print(time.time() - t0)
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    renderer = Renderer(pipe.wilor_model.mano.faces)

    render_image = image.copy()
    render_image = render_image.astype(np.float32)[:, :, ::-1] / 255.0

    for i, out in enumerate(outputs):
        verts = out["wilor_preds"]['pred_vertices'][0]
        joints = out["wilor_preds"]['pred_keypoints_3d'][0]
        is_right = out['is_right']

        verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
        joints[:, 0] = (2 * is_right - 1) * joints[:, 0]
        cam_t = out["wilor_preds"]['pred_cam_t_full'][0]
        scaled_focal_length = out["wilor_preds"]['scaled_focal_length']

        misc_args = dict(
            mesh_base_color=LIGHT_PURPLE,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
        )
        tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_PURPLE, is_right=is_right)
        tmesh.export(os.path.join(save_dir, f'{os.path.basename(img_path)}_hand{i:02d}.obj'))
        cam_view = renderer.render_rgba(verts, cam_t=cam_t, render_res=[image.shape[1], image.shape[0]],
                                        is_right=is_right,
                                        **misc_args)

        # Overlay image
        render_image = render_image[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
    render_image = (255 * render_image).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), render_image)
    print(os.path.join(save_dir, os.path.basename(img_path)))


def test_wilor_video_pipeline():
    import cv2
    import torch
    import numpy as np
    import os
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

    LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    video_path = "assets/video.mp4"
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    renderer = Renderer(pipe.wilor_model.mano.faces)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object
    output_path = os.path.join(save_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        outputs = pipe.predict(image)
        print(time.time() - t0)
        render_image = image.copy()
        render_image = render_image.astype(np.float32)[:, :, ::-1] / 255.0

        for i, out in enumerate(outputs):
            verts = out["wilor_preds"]['pred_vertices'][0]
            joints = out["wilor_preds"]['pred_keypoints_3d'][0]
            is_right = out['is_right']

            verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
            joints[:, 0] = (2 * is_right - 1) * joints[:, 0]
            cam_t = out["wilor_preds"]['pred_cam_t_full'][0]
            scaled_focal_length = out["wilor_preds"]['scaled_focal_length']

            misc_args = dict(
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            # tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_PURPLE, is_right=is_right)
            cam_view = renderer.render_rgba(verts, cam_t=cam_t, render_res=[image.shape[1], image.shape[0]],
                                            is_right=is_right,
                                            **misc_args)

            # Overlay image
            render_image = render_image[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

        render_image = (255 * render_image).astype(np.uint8)

        # Write the frame to the output video
        vout.write(render_image)

        frame_count += 1
        print(f"Processed frame {frame_count}")

    # Release everything
    cap.release()
    vout.release()
    cv2.destroyAllWindows()

    print(f"Video processing complete. Output saved to {output_path}")

def test_mediapipe_image_pipeline():
    # @markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.
    import cv2
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    import numpy as np

    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

    def draw_landmarks_on_image(rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image

    # STEP 1: Import the necessary modules.
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    # STEP 2: Create an HandLandmarker object.
    base_options = python.BaseOptions(model_asset_path='lerobot\scripts\hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                           num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    img_path = "lerobot/scripts/img.png"
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # STEP 4: Detect hand landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the classification result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv2.imshow("test", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey()

def test_apriltag(family, img_name, show_image=True):
    import cv2
    import numpy as np
    from pupil_apriltags import Detector

    # Initialize the AprilTag detector with tag16h5
    at_detector = Detector(families=family, nthreads=8, quad_decimate=2.0,quad_sigma=0.0)

    # Load the grayscale image where the tags are located
    image_path = f'lerobot/scripts/{img_name}'  # Image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #image = cv2.resize(image, (0, 0), fx=0.75, fy=0.75)

    # Camera parameters (fx, fy, cx, cy) - example values, replace with your calibration values
    cx, cy = (image.shape[1] / 2, image.shape[0] / 2)
    camera_params = (600.0, 600.0, cx, cy)  # Example focal lengths and principal point
    tag_size = 0.0156  # Tag size in meters (e.g., 2.5 cm)

    # Detect AprilTags in the image
    detections = at_detector.detect(image, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

    if show_image:
        # Convert grayscale image to color for drawing
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Define 3D axis points for visualization (1 unit length for each axis)
        axis_length = 0.05  # Length of the axis lines in meters
        axis_points = np.float32([
            [0, 0, 0],  # Origin
            [axis_length, 0, 0],  # X-axis
            [0, axis_length, 0],  # Y-axis
            [0, 0, -axis_length]  # Z-axis (negative Z for correct orientation)
        ]).reshape(-1, 3)

        # Define cube vertices relative to the tag's origin
        cube_size = 0.02  # Cube side length in meters
        half_size = cube_size / 2
        cube_points = np.float32([
            [-half_size, -half_size, 0],  # Bottom vertices
            [half_size, -half_size, 0],

            [half_size, half_size, 0],
            [-half_size, half_size, 0],
            [-half_size, -half_size, -cube_size],  # Top vertices
            [half_size, -half_size, -cube_size],
            [half_size, half_size, -cube_size],
            [-half_size, half_size, -cube_size]
        ])

        # Process each detected tag
        for detection in detections:
            # Get the rotation and translation vectors
            rvec = cv2.Rodrigues(detection.pose_R)[0]  # Convert rotation matrix to rotation vector
            tvec = detection.pose_t.flatten()

            # Project 3D axis points to the image plane
            imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, np.array([[camera_params[0], 0, camera_params[2]],
                                                                             [0, camera_params[1], camera_params[3]],
                                                                             [0, 0, 1]]),
                                          np.zeros(4))

            # Draw the axis lines on the image
            imgpts = imgpts.reshape(-1, 2)
            center = tuple(imgpts[0].astype(int))
            cv2.line(output_image, center, tuple(imgpts[1].astype(int)), (0, 0, 255), 2)  # X-axis in red
            cv2.line(output_image, center, tuple(imgpts[2].astype(int)), (0, 255, 0), 2)  # Y-axis in green
            cv2.line(output_image, center, tuple(imgpts[3].astype(int)), (255, 0, 0), 2)  # Z-axis in blue

            # Project cube vertices to the image plane
            imgpts_cube, _ = cv2.projectPoints(cube_points, rvec, tvec,
                                               np.array([[camera_params[0], 0, camera_params[2]],
                                                         [0, camera_params[1], camera_params[3]],
                                                         [0, 0, 1]]),
                                               np.zeros(4))
            imgpts_cube = imgpts_cube.reshape(-1, 2).astype(int)

            # Draw the bottom and top faces of the cube
            for i in range(4):
                cv2.line(output_image, tuple(imgpts_cube[i]), tuple(imgpts_cube[(i + 1) % 4]), (255, 255, 0),
                         2)  # Bottom
                cv2.line(output_image, tuple(imgpts_cube[i + 4]), tuple(imgpts_cube[(i + 1) % 4 + 4]), (255, 255, 0),
                         2)  # Top
                cv2.line(output_image, tuple(imgpts_cube[i]), tuple(imgpts_cube[i + 4]), (255, 255, 0), 2)  # Sides

            # Display the tag ID at the center
            cv2.putText(output_image, f"ID: {detection.tag_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        2)

        # Show and save the annotated image
        cv2.imshow("Detected AprilTags with Axes and Cube", output_image)
        cv2.imwrite("lerobot/scripts/apriltag_annotated_with_cube.png", output_image)  # Save the annotated image
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_apriltag_speed(family, img_name):
    import time

    num_runs = 25
    start_time = time.time()

    for _ in range(num_runs):
        test_apriltag(family, img_name, show_image=False)

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_run = total_time / num_runs

    print(f"Total time for {num_runs} runs: {total_time:.4f} seconds")
    print(f"Average time per run: {avg_time_per_run:.4f} seconds")


if __name__ == '__main__':
    test_apriltag(family="tag25h9", img_name="apriltag6.jpg")
    # test_wilor_image_pipeline()
    # test_wilor_video_pipeline()
    # test_mediapipe_image_pipeline()