import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def dh_transform(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1]
    ], dtype=float)


class SixDOFArm:
    def __init__(self):
        self.a =     [0.0, 0.35, 0.30, 0.0, 0.0, 0.0]
        self.alpha = [np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]
        self.d =     [0.25, 0.0, 0.0, 0.25, 0.0, 0.12]

    def forward_kinematics(self, q):
        T = np.eye(4)
        transforms = [T.copy()]
        positions = [T[:3, 3].copy()]

        for i in range(6):
            A_i = dh_transform(self.a[i], self.alpha[i], self.d[i], q[i])
            T = T @ A_i
            transforms.append(T.copy())
            positions.append(T[:3, 3].copy())

        return np.array(positions), np.array(transforms)

    def end_effector_position(self, q):
        positions, _ = self.forward_kinematics(q)
        return positions[-1]

    def jacobian_position(self, q):
        positions, transforms = self.forward_kinematics(q)
        p_end = positions[-1]

        J = np.zeros((3, 6))

        for i in range(6):
            z_i = transforms[i][:3, 2]
            p_i = transforms[i][:3, 3]
            J[:, i] = np.cross(z_i, p_end - p_i)

        return J

    def inverse_kinematics(self, target, q_init=None, max_iter=1000, tol=1e-4):
        if q_init is None:
            q = np.zeros(6)
        else:
            q = q_init.copy()

        for _ in range(max_iter):
            current = self.end_effector_position(q)
            error = target - current

            if np.linalg.norm(error) < tol:
                break

            J = self.jacobian_position(q)

            damping = 0.08
            J_inv = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(3))

            dq = J_inv @ error

            max_step = 0.05
            norm_dq = np.linalg.norm(dq)

            if norm_dq > max_step:
                dq = dq / norm_dq * max_step

            q += dq

            # Keep angles between -pi and pi
            q = (q + np.pi) % (2 * np.pi) - np.pi

        return q


def constant_speed_path(p1, p2, speed=0.01):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    distance = np.linalg.norm(p2 - p1)
    steps = max(int(distance / speed), 2)

    return np.linspace(p1, p2, steps)


def check_reachability(arm, point, name="point"):
    point = np.array(point, dtype=float)

    base_position = np.array([0.0, 0.0, 0.0])

    max_reach = sum(abs(v) for v in arm.a) + sum(abs(v) for v in arm.d)
    min_reach = arm.d[0] * 0.5

    distance = np.linalg.norm(point - base_position)

    if point[2] < 0:
        raise ValueError(
            f"{name} {point} is below ground. z must be >= 0."
        )

    if distance > max_reach:
        raise ValueError(
            f"{name} {point} is outside workspace. "
            f"Distance = {distance:.3f}, max reach = {max_reach:.3f}"
        )

    if distance < min_reach:
        raise ValueError(
            f"{name} {point} is too close to the robot base. "
            f"Distance = {distance:.3f}, min safe reach = {min_reach:.3f}"
        )

    return distance, min_reach, max_reach


def simulate_arm_move(x1, y1, z1, x2, y2, z2):
    arm = SixDOFArm()

    start = np.array([x1, y1, z1], dtype=float)
    end = np.array([x2, y2, z2], dtype=float)

    start_distance, min_reach, max_reach = check_reachability(
        arm, start, "Start point"
    )

    end_distance, _, _ = check_reachability(
        arm, end, "End point"
    )

    path = constant_speed_path(start, end, speed=0.01)

    q_list = []
    joint_positions_list = []

    q_current = np.zeros(6)

    for target in path:
        q_current = arm.inverse_kinematics(target, q_init=q_current)
        positions, _ = arm.forward_kinematics(q_current)

        q_list.append(q_current.copy())
        joint_positions_list.append(positions)

    q_tensor = np.array(q_list)
    position_tensor = np.array(joint_positions_list)

    print("Reachability check passed.")
    print("Min safe reach:", min_reach)
    print("Max reach:", max_reach)
    print("Start distance:", start_distance)
    print("End distance:", end_distance)
    print("Joint angle tensor shape:", q_tensor.shape)
    print("Position tensor shape:", position_tensor.shape)
    print("Final joint angles:")
    print(q_tensor[-1])
    print("Final end-effector position:")
    print(position_tensor[-1, -1])

    animate_robot(position_tensor, start, end)

    return q_tensor, position_tensor


def animate_robot(position_tensor, start, end):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    all_points = position_tensor.reshape(-1, 3)

    xmin, ymin, zmin = all_points.min(axis=0) - 0.1
    xmax, ymax, zmax = all_points.max(axis=0) + 0.1

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(max(0, zmin), zmax)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    line, = ax.plot([], [], [], "o-", linewidth=3)

    ax.scatter(start[0], start[1], start[2], s=80, color='green',label="Start")
    ax.scatter(end[0], end[1], end[2], s=80, color='red',label="End")
    ax.legend()

    def update(frame):
        pts = position_tensor[frame]
        line.set_data(pts[:, 0], pts[:, 1])
        line.set_3d_properties(pts[:, 2])
        return line,

    ani = FuncAnimation(
        fig,
        update,
        frames=len(position_tensor),
        interval=40,
        blit=False
    )

    plt.show()


# -----------------------------
# Safe example
# -----------------------------

q_tensor, position_tensor = simulate_arm_move(
    x1=0.25, y1=0.05, z1=0.35,
    x2=0.65, y2=-0.30, z2=0.45
)