import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# DH transformation
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
# One composable arm module
# ============================================================

class ArmModule:
    def __init__(self, a, alpha, d, radius_start, radius_end, color):
        self.a = a
        self.alpha = alpha
        self.d = d
        self.radius_start = radius_start
        self.radius_end = radius_end
        self.color = color


# ============================================================
# Compositional robotic arm
# ============================================================

class CompositionalRobotArm:
    def __init__(self):
        self.modules = []

    def add_module(self, a, alpha, d, radius_start=0.05, radius_end=0.04, color="#3498DB"):
        module = ArmModule(
            a=a,
            alpha=alpha,
            d=d,
            radius_start=radius_start,
            radius_end=radius_end,
            color=color
        )
        self.modules.append(module)

    @property
    def n(self):
        return len(self.modules)

    def forward_kinematics(self, q):
        T = np.eye(4)

        transforms = [T.copy()]
        positions = [T[:3, 3].copy()]

        for i, module in enumerate(self.modules):
            A_i = dh_transform(
                module.a,
                module.alpha,
                module.d,
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

    def inverse_kinematics(self, target, q_init=None, max_iter=1500, tol=1e-4):
        q = np.zeros(self.n) if q_init is None else q_init.copy()
        target = np.array(target, dtype=float)

        for _ in range(max_iter):
            current = self.end_effector_position(q)
            error = target - current

            if np.linalg.norm(error) < tol:
                break

            J = self.jacobian_position(q)

            damping = 0.10

            J_inv = J.T @ np.linalg.inv(
                J @ J.T + damping**2 * np.eye(3)
            )

            dq = J_inv @ error

            max_step = 0.035
            norm_dq = np.linalg.norm(dq)

            if norm_dq > max_step:
                dq = dq / norm_dq * max_step

            q += dq
            q = (q + np.pi) % (2 * np.pi) - np.pi

        return q


# ============================================================
# Smooth path
# ============================================================

def smooth_path(p1, p2, steps=220):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    t = np.linspace(0, 1, steps)

    s = 10 * t**3 - 15 * t**4 + 6 * t**5

    return np.array([p1 + si * (p2 - p1) for si in s])


# ============================================================
# Reachability check
# ============================================================

def check_reachability(arm, point, name="point"):
    point = np.array(point, dtype=float)

    max_reach = sum(abs(m.a) + abs(m.d) for m in arm.modules)
    min_reach = 0.05

    distance = np.linalg.norm(point)

    if point[2] < 0:
        raise ValueError(f"{name} {point} is below ground.")

    if distance > max_reach:
        raise ValueError(
            f"{name} {point} is outside approximate workspace. "
            f"Distance={distance:.3f}, max reach={max_reach:.3f}"
        )

    if distance < min_reach:
        raise ValueError(
            f"{name} {point} is too close to base. "
            f"Distance={distance:.3f}"
        )

    return distance, min_reach, max_reach


# ============================================================
# Tapered link drawing
# ============================================================

def plot_tapered_link(ax, p1, p2, r1, r2, color="gray", segments=24):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    v = p2 - p1
    length = np.linalg.norm(v)

    if length < 1e-8:
        return []

    v = v / length

    temp = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(v, temp)) > 0.9:
        temp = np.array([0.0, 1.0, 0.0])

    n1 = np.cross(v, temp)
    n1 = n1 / np.linalg.norm(n1)

    n2 = np.cross(v, n1)

    theta = np.linspace(0, 2 * np.pi, segments)

    circle1 = np.array([
        p1 + r1 * (np.cos(t) * n1 + np.sin(t) * n2)
        for t in theta
    ])

    circle2 = np.array([
        p2 + r2 * (np.cos(t) * n1 + np.sin(t) * n2)
        for t in theta
    ])

    X = np.vstack([circle1[:, 0], circle2[:, 0]])
    Y = np.vstack([circle1[:, 1], circle2[:, 1]])
    Z = np.vstack([circle1[:, 2], circle2[:, 2]])

    surface = ax.plot_surface(
        X,
        Y,
        Z,
        color=color,
        alpha=0.95,
        linewidth=0,
        shade=True
    )

    return [surface]


# ============================================================
# Animation
# ============================================================

def animate_robot(arm, position_tensor, start, end, interval=20):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    all_points = np.vstack([
        position_tensor.reshape(-1, 3),
        start.reshape(1, 3),
        end.reshape(1, 3)
    ])

    xmin, ymin, zmin = all_points.min(axis=0) - 0.15
    xmax, ymax, zmax = all_points.max(axis=0) + 0.15

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(max(0, zmin), zmax)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Compositional Robotic Arm with {arm.n} Modules")

    ax.scatter(start[0], start[1], start[2], s=120, color="green", label="Start")
    ax.scatter(end[0], end[1], end[2], s=120, color="red", label="End")

    trajectory = position_tensor[:, -1, :]

    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        linewidth=2,
        color="blue",
        label="End-effector path"
    )

    current_objects = []

    def update(frame):
        nonlocal current_objects

        for obj in current_objects:
            obj.remove()

        current_objects = []

        pts = position_tensor[frame]

        for i in range(len(pts) - 1):
            module = arm.modules[i]

            surfaces = plot_tapered_link(
                ax,
                pts[i],
                pts[i + 1],
                module.radius_start,
                module.radius_end,
                color=module.color
            )

            current_objects.extend(surfaces)

        joints = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=45,
            color="#ECEFF1"
        )

        current_objects.append(joints)

        return current_objects

    ani = FuncAnimation(
        fig,
        update,
        frames=len(position_tensor),
        interval=interval,
        blit=False
    )

    ax.legend()
    plt.show()


# ============================================================
# Simulation
# ============================================================

def simulate_arm_move(
    arm,
    x1, y1, z1,
    x2, y2, z2,
    steps=220,
    interval=20
):
    start = np.array([x1, y1, z1], dtype=float)
    end = np.array([x2, y2, z2], dtype=float)

    check_reachability(arm, start, "Start point")
    check_reachability(arm, end, "End point")

    path = smooth_path(start, end, steps=steps)

    q_list = []
    position_list = []
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
        position_list.append(positions)
        error_list.append(error)
        rank_list.append(rank)

    q_tensor = np.array(q_list)
    position_tensor = np.array(position_list)
    error_tensor = np.array(error_list)
    rank_tensor = np.array(rank_list)

    print("Number of modules:", arm.n)
    print("Joint tensor shape:", q_tensor.shape)
    print("Position tensor shape:", position_tensor.shape)
    print("Mean IK error:", error_tensor.mean())
    print("Max IK error:", error_tensor.max())
    print("Minimum Jacobian rank:", rank_tensor.min())
    print("Final joint angles:")
    print(q_tensor[-1])
    print("Final end-effector position:")
    print(position_tensor[-1, -1])

    animate_robot(
        arm,
        position_tensor,
        start,
        end,
        interval=interval
    )

    return q_tensor, position_tensor, error_tensor, rank_tensor


# ============================================================
# Build robot compositionally
# ============================================================

arm = CompositionalRobotArm()

# Base module
arm.add_module(
    a=0.0,
    alpha=np.pi / 2,
    d=0.25,
    radius_start=0.070,
    radius_end=0.055,
    color="#0B132B"
)

# Shoulder module
arm.add_module(
    a=0.35,
    alpha=0.0,
    d=0.0,
    radius_start=0.055,
    radius_end=0.045,
    color="#1C2541"
)

#Elbow module
arm.add_module(
    a=0.30,
    alpha=0.0,
    d=0.0,
    radius_start=0.045,
    radius_end=0.035,
    color="#3A506B"
)

# Wrist module 1
arm.add_module(
    a=0.0,
    alpha=np.pi / 2,
    d=0.25,
    radius_start=0.035,
    radius_end=0.027,
    color="#5BC0BE"
)

# Wrist module 2
arm.add_module(
    a=0.0,
    alpha=-np.pi / 2,
    d=0.0,
    radius_start=0.027,
    radius_end=0.020,
    color="#C5C6C7"
)
#
# End module
arm.add_module(
    a=0.0,
    alpha=0.0,
    d=0.12,
    radius_start=0.020,
    radius_end=0.014,
    color="#F5F7FA"
)


# ============================================================
# Add more modules forever if you want
# ============================================================

# Example extra module:
arm.add_module(
    a=0.20,
    alpha=0.0,
    d=0.0,
    radius_start=0.014,
    radius_end=0.010,
    color="#FFB703"
)

#arm.add_module(
#    a=0.20,
#    alpha=0.0,
#    d=0.0,
#    radius_start=0.014,
#    radius_end=0.010,
#    color="#FFB704"
#)
#
#
#arm.add_module(
#    a=0.20,
#    alpha=0.0,
#    d=0.0,
#    radius_start=0.014,
#    radius_end=0.010,
#    color="#FFB705"
#)



# ============================================================
# Run simulation
# ============================================================

q_tensor, position_tensor, error_tensor, rank_tensor = simulate_arm_move(
    arm,
    x1=0.25,
    y1=0.05,
    z1=0.35,
    x2=0.65,
    y2=0.30,
    z2=0.45,
    steps=220,
    interval=20
)