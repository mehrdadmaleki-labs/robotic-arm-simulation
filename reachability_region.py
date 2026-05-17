import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# Homogeneous transformation tensor using DH parameters
# ============================================================

def dh_transform(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1]
    ], dtype=float)


# ============================================================
# 6-DOF Robotic Arm
# ============================================================

class SixDOFArm:
    def __init__(self):
        self.a =     [0.0, 0.35, 0.30, 0.0, 0.0, 0.0]
        self.alpha = [np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]
        self.d =     [0.25, 0.0, 0.0, 0.25, 0.0, 0.12]

        self.n = len(self.a)

    def forward_kinematics(self, q):
        T = np.eye(4)

        transforms = [T.copy()]
        positions = [T[:3, 3].copy()]

        for i in range(self.n):
            A_i = dh_transform(
                self.a[i],
                self.alpha[i],
                self.d[i],
                q[i]
            )

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
        J = np.zeros((3, self.n))

        for i in range(self.n):
            z_i = transforms[i][:3, 2]
            p_i = transforms[i][:3, 3]

            J[:, i] = np.cross(z_i, p_end - p_i)

        return J

    def inverse_kinematics(self, target, q_init=None, max_iter=1000, tol=1e-4):
        if q_init is None:
            q = np.zeros(self.n)
        else:
            q = q_init.copy()

        target = np.array(target, dtype=float)

        for _ in range(max_iter):
            current = self.end_effector_position(q)
            error = target - current

            if np.linalg.norm(error) < tol:
                break

            J = self.jacobian_position(q)

            damping = 0.08
            J_inv = J.T @ np.linalg.inv(
                J @ J.T + damping**2 * np.eye(3)
            )

            dq = J_inv @ error

            max_step = 0.05
            norm_dq = np.linalg.norm(dq)

            if norm_dq > max_step:
                dq = dq / norm_dq * max_step

            q += dq

            q = (q + np.pi) % (2 * np.pi) - np.pi

        return q


# ============================================================
# Path generation
# ============================================================

def constant_speed_path(p1, p2, speed=0.01):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    distance = np.linalg.norm(p2 - p1)
    steps = max(int(distance / speed), 2)

    return np.linspace(p1, p2, steps)


# ============================================================
# Reachability check
# ============================================================

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


# ============================================================
# Workspace computation
# ============================================================

def compute_workspace_monte_carlo(arm, samples=50000):
    points = []

    for _ in range(samples):
        q = np.random.uniform(
            low=-np.pi,
            high=np.pi,
            size=arm.n
        )

        p = arm.end_effector_position(q)
        points.append(p)

    return np.array(points)


def compute_regular_workspace_monte_carlo(
    arm,
    samples=50000,
    rank_tol=1e-5
):
    points = []
    singular_points = []

    for _ in range(samples):
        q = np.random.uniform(
            low=-np.pi,
            high=np.pi,
            size=arm.n
        )

        p = arm.end_effector_position(q)
        Jp = arm.jacobian_position(q)

        rank = np.linalg.matrix_rank(Jp, tol=rank_tol)

        if rank == 3:
            points.append(p)
        else:
            singular_points.append(p)

    return np.array(points), np.array(singular_points)


# ============================================================
# Plot workspace only
# ============================================================

def plot_workspace(workspace_points):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        workspace_points[:, 0],
        workspace_points[:, 1],
        workspace_points[:, 2],
        s=1,
        alpha=0.25
    )

    ax.set_title("Reachable Workspace")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


# ============================================================
# Animation without workspace
# ============================================================

def animate_robot(position_tensor, start, end):
    fig = plt.figure(figsize=(10, 8))
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

    ax.scatter(start[0], start[1], start[2], s=80, label="Start")
    ax.scatter(end[0], end[1], end[2], s=80, label="End")

    line, = ax.plot([], [], [], "o-", linewidth=3)

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


# ============================================================
# Workspace + animated robot together
# ============================================================

def visualize_workspace_and_motion(
    workspace_points,
    position_tensor,
    start,
    end,
    singular_points=None
):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        workspace_points[:, 0],
        workspace_points[:, 1],
        workspace_points[:, 2],
        s=1,
        alpha=0.08,
        label="Regular reachable workspace"
    )

    if singular_points is not None and len(singular_points) > 0:
        ax.scatter(
            singular_points[:, 0],
            singular_points[:, 1],
            singular_points[:, 2],
            s=2,
            alpha=0.25,
            label="Singular samples"
        )

    ax.scatter(start[0], start[1], start[2], s=120, label="Start")
    ax.scatter(end[0], end[1], end[2], s=120, label="End")

    trajectory = position_tensor[:, -1, :]

    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        linewidth=2,
        label="End-effector path"
    )

    line, = ax.plot([], [], [], "o-", linewidth=3, label="Robot arm")

    all_points = np.vstack([
        workspace_points,
        position_tensor.reshape(-1, 3),
        start.reshape(1, 3),
        end.reshape(1, 3)
    ])

    xmin, ymin, zmin = all_points.min(axis=0) - 0.1
    xmax, ymax, zmax = all_points.max(axis=0) + 0.1

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(max(0, zmin), zmax)

    ax.set_title("6-DOF Robotic Arm: Reachable Workspace + Motion")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

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


# ============================================================
# Main simulation function
# ============================================================

def simulate_arm_move(
    x1, y1, z1,
    x2, y2, z2,
    show_workspace=True,
    workspace_samples=50000,
    show_singular_samples=False
):
    arm = SixDOFArm()

    start = np.array([x1, y1, z1], dtype=float)
    end = np.array([x2, y2, z2], dtype=float)

    start_distance, min_reach, max_reach = check_reachability(
        arm,
        start,
        "Start point"
    )

    end_distance, _, _ = check_reachability(
        arm,
        end,
        "End point"
    )

    path = constant_speed_path(start, end, speed=0.01)

    q_list = []
    joint_positions_list = []
    error_list = []
    rank_list = []

    q_current = np.zeros(arm.n)

    for target in path:
        q_current = arm.inverse_kinematics(
            target,
            q_init=q_current
        )

        positions, _ = arm.forward_kinematics(q_current)

        actual = positions[-1]
        error = np.linalg.norm(target - actual)

        Jp = arm.jacobian_position(q_current)
        rank = np.linalg.matrix_rank(Jp)

        q_list.append(q_current.copy())
        joint_positions_list.append(positions)
        error_list.append(error)
        rank_list.append(rank)

    q_tensor = np.array(q_list)
    position_tensor = np.array(joint_positions_list)
    error_tensor = np.array(error_list)
    rank_tensor = np.array(rank_list)

    print("Reachability check passed.")
    print("Min safe reach:", min_reach)
    print("Max reach:", max_reach)
    print("Start distance:", start_distance)
    print("End distance:", end_distance)

    print("Joint angle tensor shape:", q_tensor.shape)
    print("Position tensor shape:", position_tensor.shape)

    print("Mean IK error:", error_tensor.mean())
    print("Max IK error:", error_tensor.max())
    print("Minimum Jacobian rank along path:", rank_tensor.min())

    print("Final joint angles:")
    print(q_tensor[-1])

    print("Final end-effector position:")
    print(position_tensor[-1, -1])

    if show_workspace:
        print("Computing reachable workspace...")

        if show_singular_samples:
            workspace_points, singular_points = compute_regular_workspace_monte_carlo(
                arm,
                samples=workspace_samples
            )
        else:
            workspace_points = compute_workspace_monte_carlo(
                arm,
                samples=workspace_samples
            )
            singular_points = None

        visualize_workspace_and_motion(
            workspace_points,
            position_tensor,
            start,
            end,
            singular_points=singular_points
        )

    else:
        animate_robot(
            position_tensor,
            start,
            end
        )

    return q_tensor, position_tensor, error_tensor, rank_tensor


# ============================================================
# Example
# ============================================================

q_tensor, position_tensor, error_tensor, rank_tensor = simulate_arm_move(
    x1=0.25,
    y1=0.05,
    z1=0.35,

    x2=0.65,
    y2=0.30,
    z2=0.45,

    show_workspace=True,
    workspace_samples=40000,
    show_singular_samples=False
)