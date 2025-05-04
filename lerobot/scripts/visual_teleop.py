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
import cv2


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

def project_and_draw_3d_keypoints(image_bgr, keypoints3d, cam_t=None, focal=None, color=(0, 255, 255)):
    """
    Projects 3D keypoints onto the 2D image plane and draws them.

    Args:
        image_bgr   (np.ndarray): BGR image to draw on.
        keypoints3d (np.ndarray): (N, 3) 3D keypoints.
        cam_t       (np.ndarray): (3,) camera translation vector.
        focal       (float): focal length (same for fx, fy).
        color       (tuple): BGR color to draw points.
    """
    h, w = image_bgr.shape[:2]
    if focal is None:
        focal = 12500
    if cam_t is None:
        cam_t = np.zeros(3)

    fx = fy = focal
    cx, cy = w / 2, h / 2
    X = keypoints3d[:, 0] + cam_t[0]
    Y = keypoints3d[:, 1] + cam_t[1]
    Z = keypoints3d[:, 2] + cam_t[2] + 1e-6

    xs = fx * X / Z + cx
    ys = fy * Y / Z + cy
    pts2d = np.stack([xs, ys], axis=1).astype(int)

    for x, y in pts2d:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image_bgr, (x, y), 4, color, -1)

    return image_bgr

def test_wilor(cam_idx=0, full_render=True):
    """
    Live demo for WiLor hand‑pose.

    Args
    ----
    cam_idx (int | str): 0 for default webcam, or path / RTSP URL.
    full_render (bool):  True  → render full mesh with Renderer  
                         False → skip mesh and just overlay 2‑D projected joints
    """
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

    LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    pipe     = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float16, verbose=False)
    renderer = Renderer(pipe.wilor_model.mano.faces) if full_render else None
    cap      = cv2.VideoCapture(cam_idx)

    smoothed_fps = None
    alpha = 0.1  # smoothing factor

    while cap.isOpened():
        t0 = time.time()
        ok, frame_bgr = cap.read()
        if not ok: break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        outputs   = pipe.predict(frame_rgb)

        vis = frame_bgr.astype(np.float32)[:, :, ::-1] / 255.0  # RGB float32

        for out in outputs:
            verts   = out["wilor_preds"]["pred_vertices"][0]
            joints3 = out["wilor_preds"]["pred_keypoints_3d"][0]
            cam_t   = out["wilor_preds"]["pred_cam_t_full"][0]
            focal   = out["wilor_preds"]["scaled_focal_length"]
            right   = out["is_right"]

            if right == 0:
                verts[:, 0]  *= -1
                joints3[:, 0] *= -1

            if full_render:
                rgba = renderer.render_rgba(
                    verts,
                    cam_t=cam_t,
                    render_res=[frame_rgb.shape[1], frame_rgb.shape[0]],
                    is_right=right,
                    mesh_base_color=LIGHT_PURPLE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=focal,
                )
                vis = vis[:, :, :3] * (1 - rgba[:, :, 3:]) + rgba[:, :, :3] * rgba[:, :, 3:]
            else:
                 frame_bgr = project_and_draw_3d_keypoints(
                    frame_bgr,
                    keypoints3d=joints3,
                    cam_t=cam_t,
                    focal=focal,
                    color=(255, 0, 255)  # magenta
                )

        # ------------- compute smoothed FPS & display -------------
        fps = 1.0 / (time.time() - t0)
        smoothed_fps = fps if smoothed_fps is None else (1 - alpha) * smoothed_fps + alpha * fps

        cv2.putText(frame_bgr, f"{smoothed_fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out_bgr = (vis[:, :, ::-1] * 255).astype(np.uint8) if full_render else frame_bgr
        cv2.imshow("WiLor Hand Pose 3D", out_bgr)

        if cv2.waitKey(1) in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

def test_wilor_realtime_3d(
        cam_idx: int | str = 0,
        detect_every: int = 10,
        lk_win: int = 15,
        lk_levels: int = 3,
        pnp_fail_thresh: int = 10,
):
    """
    Real-time 3-D hand-pose tracking with WiLor + LK optical flow + solvePnP.
    • FPS overlay
    • survives hand entering/exiting frame
    • uses your exact 2D projection formula
    """
    import cv2, time
    import torch, numpy as np
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    # ── init ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe   = WiLorHandPose3dEstimationPipeline(device=device,
                                               dtype=torch.float16,
                                               verbose=False)
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Couldn’t open camera/stream: {cam_idx}")
    cv2.namedWindow("WiLor 3D tracking", cv2.WINDOW_NORMAL)

    lk_params = dict(
        winSize=(lk_win, lk_win),
        maxLevel=lk_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )

    # ── persistent state ─────────────────────────────────────────────────
    P_obj = None       # (21,3) local joint coords
    fx = fy = cx = cy = None
    R = None           # current cam rotation
    t = None           # current cam translation
    pts2d_prev = None
    grey_prev = None
    frame_idx = 0
    fps, alpha = 0.0, 0.9
    t_last = time.time()

    while cap.isOpened():
        ok, frame_bgr = cap.read()
        if not ok:
            break
        grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = grey.shape

        need_detect = (
            P_obj is None
            or pts2d_prev is None
            or (frame_idx % detect_every == 0)
        )

        # ── heavy pass (full WiLor infer + init state) ───────────────────
        if need_detect:
            outs = pipe.predict(frame_bgr[..., ::-1])  # BGR→RGB
            if outs:
                out      = outs[0]
                joints3  = out["wilor_preds"]["pred_keypoints_3d"][0]  # (21,3)
                cam_t    = out["wilor_preds"]["pred_cam_t_full"][0]    # (3,)
                # object-local coords
                P_obj    = joints3 - joints3[0]
                # intrinsics
                fx = fy = out["wilor_preds"]["scaled_focal_length"]
                cx, cy  = w/2, h/2
                # initial cam pose
                R = np.eye(3, dtype=np.float32)
                t = cam_t.reshape(3,1)
                # project exactly as your demo did:
                u = fx * (joints3[:,0] + cam_t[0]) / (joints3[:,2] + cam_t[2]) + cx
                v = fy * (joints3[:,1] + cam_t[1]) / (joints3[:,2] + cam_t[2]) + cy
                pts2d = np.stack([u, v], axis=1)
                pts2d_prev = pts2d.astype(np.float32)
                grey_prev   = grey.copy()
            else:
                # no detection → just display
                cv2.putText(frame_bgr, "No hand",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
                cv2.imshow("WiLor 3D tracking", frame_bgr)
                if cv2.waitKey(1) in (27, ord('q')):
                    break
                frame_idx += 1
                continue

        # ── light pass (LK flow + solvePnP) ──────────────────────────────
        else:
            pts, st, _ = cv2.calcOpticalFlowPyrLK(
                grey_prev, grey,
                pts2d_prev.reshape(-1,1,2),
                None, **lk_params
            )
            pts2d = pts.reshape(-1,2)

            # if too many lost → force re-detect next loop
            if st.sum() < pnp_fail_thresh:
                pts2d_prev = None
                grey_prev   = grey
                frame_idx  += 1
                continue

            # solvePnP to get new R,t
            K = np.array([[fx, 0,  cx],
                          [0,  fy, cy],
                          [0,   0,  1]], dtype=np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                P_obj, pts2d, K, None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                pts2d_prev = None
                grey_prev   = grey
                frame_idx  += 1
                continue

            R, _ = cv2.Rodrigues(rvec)
            t     = tvec
            # reconstruct camera-space joints
            joints_cam = (R @ P_obj.T + t).T  # (21,3)
            # project again with your exact formula
            u = fx * joints_cam[:,0] / joints_cam[:,2] + cx
            v = fy * joints_cam[:,1] / joints_cam[:,2] + cy
            pts2d = np.stack([u, v], axis=1)

        # ── draw 2D keypoints ─────────────────────────────────────────────
        for x, y in pts2d.astype(int):
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame_bgr, (x, y), 3, (255, 0, 255), -1)

        # ── FPS overlay ─────────────────────────────────────────────────
        now = time.time()
        dt  = now - t_last
        t_last = now
        fps = alpha * fps + (1 - alpha) * (1 / dt)
        cv2.putText(frame_bgr,
                    f"{fps:4.1f} FPS | detect every {detect_every}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        cv2.imshow("WiLor 3D tracking", frame_bgr)
        if cv2.waitKey(1) in (27, ord('q')):
            break

        # ── update state ────────────────────────────────────────────────
        grey_prev   = grey
        pts2d_prev  = pts2d.astype(np.float32)
        frame_idx  += 1

    cap.release()
    cv2.destroyAllWindows()

def test_mediapipe(cam_idx=0):
    import cv2
    import time
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    model_path = 'models/hand_landmarker.task'

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path,
            delegate=BaseOptions.Delegate.GPU
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
    )

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    with HandLandmarker.create_from_options(options) as landmarker:
        prev_time = time.time()
        frame_timestamp_ms = 0
        frame_interval_ms = 33  # Simulate ~30 FPS

        while True:
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += frame_interval_ms

            # Draw landmarks
            for hand_landmarks in result.hand_landmarks:
                for landmark in hand_landmarks:
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # FPS
            new_time = time.time()
            fps = 1.0 / (new_time - prev_time)
            prev_time = new_time
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("Hand Landmarks", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def test_mediapipe_gesture(cam_idx=0):
    import cv2
    import time
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    model_path = "models/gesture_recognizer.task"

    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = vision.GestureRecognizer
    GestureRecognizerOptions = vision.GestureRecognizerOptions
    VisionRunningMode = vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path,
            # To use GPU: uncomment this line
            # delegate=BaseOptions.Delegate.GPU,
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
    )

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    with GestureRecognizer.create_from_options(options) as recognizer:
        prev_time = time.time()
        timestamp_ms = 0
        interval_ms = 33  # ~30 FPS

        while True:
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = recognizer.recognize_for_video(mp_image, timestamp_ms)
            timestamp_ms += interval_ms

            # Draw landmarks and gestures
            for idx, landmarks in enumerate(result.hand_landmarks):
                for lm in landmarks:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            for idx, gesture in enumerate(result.gestures):
                if gesture:
                    name = gesture[0].category_name
                    score = gesture[0].score
                    cv2.putText(frame, f"{name} ({score:.2f})", (10, 70 + idx * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show FPS
            new_time = time.time()
            fps = 1.0 / (new_time - prev_time)
            prev_time = new_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("Gesture Recognizer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

def test_hand_detector(cam_idx=0, alpha=0.1):
    import cv2
    import time
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download("warmshao/WiLoR-mini", "pretrained_models/detector.pt")
    hand_detector = YOLO(model_path).to("cuda")
    # hand_detector = YOLO(r"C:\github_personal\lerobot\hand_runs\hand_yolo11n4\weights\best.pt")

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError("Couldn't open camera")

    ema_fps = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Measure prediction-only time
        t0 = time.time()
        detections = hand_detector(frame_rgb, conf=0.3, verbose=False)[0]
        t1 = time.time()

        # Update EMA FPS
        fps = 1.0 / (t1 - t0)
        ema_fps = fps if ema_fps is None else alpha * fps + (1 - alpha) * ema_fps

        for det in detections:
            x1, y1, x2, y2 = map(int, det.boxes.xyxy[0])
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display smoothed FPS
        cv2.putText(frame_bgr, f"Predict FPS: {ema_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("WILOR Hand Detector", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


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
    cam_idx = 1
    # test_apriltag(family="tag25h9", img_name="apriltag6.jpg")
    # test_wilor_image_pipeline()
    test_wilor(cam_idx=cam_idx, full_render=False)
    # test_wilor_realtime_3d(cam_idx=cam_idx, detect_every=2)
    # test_mediapipe(cam_idx=cam_idx)
    # test_mediapipe_gesture(cam_idx=cam_idx)