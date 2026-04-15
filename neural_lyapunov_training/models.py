# Definitions of dynamics (inverted pendulum, quadrotor2d), lyapunov function
# and loss function.
# Everything needs to be a subclass of nn.Module in order to be handled by
# auto_LiRPA.

import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
import torch
import scipy
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../torchdyn")

import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.pendulum as pendulum
import neural_lyapunov_training.quadrotor2d as quadrotor2d
import neural_lyapunov_training.path_tracking as path_tracking
import neural_lyapunov_training.pvtol as pvtol
import neural_lyapunov_training.supply_rate as supply_rate

from neural_lyapunov_training.controllers import LinearController


class Dynamics(nn.Module):
    """
    Base class for any dynamics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, u):
        raise NotImplementedError

    @property
    def x_equilibrium(self):
        raise NotImplementedError

    @property
    def u_equilibrium(self):
        raise NotImplementedError


class CartPoleDynamics(Dynamics):
    """
    The dynamics of a cart-pole with state x = [px, θ, px_dot, θdot]
    """

    def __init__(self, mc=10.0, mp=1.0, l=1.0, gravity=9.81, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc = mc
        self.mp = mp
        self.l = l
        self.gravity = gravity

    def forward(self, x, u):
        """
        Refer to https://underactuated.mit.edu/acrobot.html#cart_pole
        """
        px_dot = x[:, 2]
        theta_dot = x[:, 3]
        s = torch.sin(x[:, 1])
        c = torch.cos(x[:, 1])
        px_ddot = (
            u.squeeze(1) + self.mp * s * (self.l * theta_dot**2 + self.gravity * c)
        ) / (self.mp * s**2 + self.mc)
        theta_ddot = (
            -u.squeeze(1) * c
            - self.mp * self.l * theta_dot**2 * c * s
            - (self.mc + self.mp) * self.gravity * s
        ) / (self.l * self.mc + self.mp * s**2)
        return torch.cat(
            (
                px_dot.unsqueeze(1),
                theta_dot.unsqueeze(1),
                px_ddot.unsqueeze(1),
                theta_ddot.unsqueeze(1),
            ),
            dim=1,
        )


class AcrobotDynamics(Dynamics):
    """
    The dynamics of an Acrobot with state x = [θ1, θ2, θdot1, θdot2]
    """

    def __init__(
        self,
        m1=1.0,
        m2=1.0,
        l1=1.0,
        l2=2.0,
        lc1=0.5,
        lc2=1.0,
        Ic1=0.083,
        Ic2=0.33,
        b1=0.1,
        b2=0.1,
        gravity=9.81,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.lc1 = lc1
        self.lc2 = lc2
        self.Ic1 = Ic1
        self.Ic2 = Ic2
        self.b1 = b1
        self.b2 = b2
        self.gravity = gravity

        # We compute the minimal value of the mass matrix determinant. This is
        # useful later when we compute the division over mass
        # matrix determinant in the dynamics.
        # The determinant is Ic1*Ic2 + Ic1*m2*lc2*lc2 + Ic2*m1*lc1*lc1 +
        # Ic2*m2*l1*l1 + m1*m2*lc1*lc1*lc2*lc2
        # +m2*m2*lc2*lc2*l1*l1*sin(theta2)*sin(theta2)
        # We can safely ignore the last term (which is always non-negative) to
        # compute the minimal.
        self.det_M_minimal = (
            self.Ic1 * self.Ic2
            + self.Ic1 * self.m2 * self.lc2**2
            + self.Ic2 * self.m1 * self.lc1**2
            + self.Ic2 * self.m2 * self.l1**2
            + self.m1 * self.m2 * self.lc1**2 * self.lc2**2
        )

    def forward(self, x, u):
        """
        Compute the continuous-time dynamics.
        The dynamics is copied from
        https://github.com/RobotLocomotion/drake/blob/master/examples/acrobot/acrobot_plant.cc
        """
        assert x.shape[0] == u.shape[0]

        # The dynamics is M * vdot = B*u - bias
        c2 = torch.cos(x[:, 1])
        I1 = self.Ic1 + self.m1 * self.lc1 * self.lc1
        I2 = self.Ic2 + self.m2 * self.lc2 * self.lc2
        m2l1lc2 = self.m2 * self.l1 * self.lc2
        m12 = I2 + m2l1lc2 * c2

        M00 = I1 + I2 + self.m2 * self.l1 * self.l1 + 2 * m2l1lc2 * c2
        M01 = m12
        M10 = m12
        M11 = I2

        # C(q, v) * v terms.
        s1 = torch.sin(x[:, 0])
        s2 = torch.sin(x[:, 1])
        s12 = torch.sin(x[:, 0] + x[:, 1])
        bias0 = (
            -2 * m2l1lc2 * s2 * x[:, 3] * x[:, 2] + -m2l1lc2 * s2 * x[:, 3] * x[:, 3]
        )
        bias1 = m2l1lc2 * s2 * x[:, 2] * x[:, 2]
        # -gravity(q) terms.
        bias0 += self.gravity * self.m1 * self.lc1 * s1 + self.gravity * self.m2 * (
            self.l1 * s1 + self.lc2 * s12
        )
        bias1 += self.gravity * self.m2 * self.lc2 * s12
        # damping terms.
        bias0 += self.b1 * x[:, 2]
        bias1 += self.b2 * x[:, 3]

        # Compute rhs = B*u - bias
        rhs0 = -bias0
        rhs1 = u.squeeze(1) - bias1

        # Solve M * vdot = rhs as vdot = M_adj * rhs / det(M)
        # To hint the verifier that det_M is strictly positive, we use ReLU to
        # bound it from below.
        det_M = (
            torch.nn.functional.relu(M00 * M11 - M01 * M10 - self.det_M_minimal)
            + self.det_M_minimal
        )
        M_adj00 = M11
        M_adj01 = -M10
        M_adj10 = -M01
        M_adj11 = M00
        vdot0 = (M_adj00 * rhs0 + M_adj01 * rhs1) / det_M
        vdot1 = (M_adj10 * rhs0 + M_adj11 * rhs1) / det_M
        v = x[:, 2:]
        return torch.cat((v, vdot0.unsqueeze(-1), vdot1.unsqueeze(-1)), dim=1)

    @property
    def x_equilibrium(self):
        return torch.zeros((4,))

    @property
    def u_equilibrium(self):
        return torch.zeros((1,))


class Quadrotor3DDynamics(Dynamics):
    """
    3D Quadrotor dyanamics, based on https://github.com/StanfordASL/neural-network-lyapunov/blob/master/neural_network_lyapunov/examples/quadrotor3d/quadrotor.py
    Modified to support batch computation and auto_LiRPA bounding.
    """

    def __init__(self, dtype):
        """
        The parameter of this quadrotor is obtained from
        Attitude stabilization of a VTOL quadrotor aircraft
        by Abdelhamid Tayebi and Stephen McGilvray.
        """
        super().__init__()
        self.mass = 0.468
        self.gravity = 9.81
        self.arm_length = 0.225
        # The inertia matrix is diagonal, we only store Ixx, Iyy and Izz.
        self.inertia = torch.tensor([4.9e-3, 4.9e-3, 8.8e-3], dtype=dtype)
        # The ratio between the torque along the z axis versus the force.
        self.z_torque_to_force_factor = 1.1 / 29
        self.dtype = dtype
        self.hover_thrust = self.mass * self.gravity / 4

    def rpy2rotmat(self, rpy):
        rpy_0 = rpy[:, 0:1]
        rpy_1 = rpy[:, 1:2]
        rpy_2 = rpy[:, 2:3]
        cos_roll = torch.cos(rpy_0)
        sin_roll = torch.sin(rpy_0)
        cos_pitch = torch.cos(rpy_1)
        sin_pitch = torch.sin(rpy_1)
        cos_yaw = torch.cos(rpy_2)
        sin_yaw = torch.sin(rpy_2)

        # Return a 3x3 tuple.
        results = (
            (
                cos_pitch * cos_yaw,
                -cos_roll * sin_yaw + cos_yaw * sin_pitch * sin_roll,
                cos_roll * cos_yaw * sin_pitch + sin_roll * sin_yaw,
            ),
            (
                cos_pitch * sin_yaw,
                cos_roll * cos_yaw + sin_pitch * sin_roll * sin_yaw,
                cos_roll * sin_pitch * sin_yaw - cos_yaw * sin_roll,
            ),
            (-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll),
        )
        return results

    def cross(self, a, b):
        # 3-d cross-product of a and b. a and b must have shape [batch, 3].
        a1 = a[:, 0:1]
        a2 = a[:, 1:2]
        a3 = a[:, 2:3]
        b1 = b[:, 0:1]
        b2 = b[:, 1:2]
        b3 = b[:, 2:3]
        s1 = a2 * b3 - a3 * b2
        s2 = a3 * b1 - a1 * b3
        s3 = a1 * b2 - a2 * b1
        return torch.cat((s1, s2, s3), dim=-1)

    def forward(self, x, u):
        # Compute the time derivative of the state.
        # The dynamics is explained in
        # Minimum Snap Trajectory Generation and Control for Quadrotors
        # by Daniel Mellinger and Vijay Kumar
        # @param x current system state, in shape [batch, 12]
        #          the states are [pos_x, pos_y, pos_z, roll, pitch, yaw,
        #          pos_xdot, pos_ydot, pos_zdot, angular_vel_x,
        #          angular_vel_y, angular_vel_z]
        # @param u the thrust generated in each propeller, its shape is
        #          [batch, 4]
        rpy = x[:, 3:6]
        pos_dot = x[:, 6:9]
        omega = x[:, 9:12]

        # plant_input is [total_thrust, torque_x, torque_y, torque_z]
        total_thrust = u.sum(dim=-1, keepdim=True)
        torque_x = self.arm_length * u[:, 1:2] - self.arm_length * u[:, 3:4]
        torque_y = -self.arm_length * u[:, 0:1] + self.arm_length * u[:, 2:3]
        torque_z = self.z_torque_to_force_factor * (
            u[:, 0:1] - u[:, 1:2] + u[:, 2:3] - u[:, 3:4]
        )
        torque = torch.cat((torque_x, torque_y, torque_z), dim=-1)

        R = self.rpy2rotmat(rpy)
        # We actually only need the last column of R.
        R_col = R[0][2], R[1][2], R[2][2]
        pos_ddot_0 = R_col[0] * total_thrust / self.mass
        pos_ddot_1 = R_col[1] * total_thrust / self.mass
        pos_ddot_2 = R_col[2] * total_thrust / self.mass - self.gravity

        # Here we exploit the fact that the inertia matrix is diagonal.
        omega_dot = (self.cross(-omega, self.inertia * omega) + torque) / self.inertia
        # Convert the angular velocity to the roll-pitch-yaw time
        # derivative.
        rpy_0 = rpy[:, 0:1]
        rpy_1 = rpy[:, 1:2]
        cos_roll = torch.cos(rpy_0)
        sin_roll = torch.sin(rpy_0)
        cos_pitch = torch.cos(rpy_1)
        tan_pitch = torch.tan(rpy_1)

        # Equation 2.7 in quadrotor control: modeling, nonlinear control
        # design and simulation by Francesco Sabatino
        omega_0 = omega[:, 0:1]
        omega_1 = omega[:, 1:2]
        omega_2 = omega[:, 2:3]
        rpy_dot_0 = (
            omega_0 + sin_roll * tan_pitch * omega_1 + cos_roll * tan_pitch * omega_2
        )
        rpy_dot_1 = cos_roll * omega_1 - sin_roll * omega_2
        rpy_dot_2 = (sin_roll / cos_pitch) * omega_1 + (cos_roll / cos_pitch) * omega_2
        return torch.cat(
            (
                pos_dot,
                rpy_dot_0,
                rpy_dot_1,
                rpy_dot_2,
                pos_ddot_0,
                pos_ddot_1,
                pos_ddot_2,
                omega_dot,
            ),
            dim=-1,
        )

    @property
    def x_equilibrium(self):
        return torch.zeros((12,), dtype=self.dtype)

    @property
    def u_equilibrium(self):
        return torch.full((4,), self.hover_thrust, dtype=self.dtype)

    def linearized_dynamics(self, x: np.ndarray, u: np.ndarray):
        """
        Return ∂ẋ/∂x and ∂ẋ/∂u
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(u, np.ndarray)
        x_torch = torch.from_numpy(x).reshape((1, -1)).to(self.inertia.device)
        u_torch = torch.from_numpy(u).reshape((1, -1)).to(self.inertia.device)
        x_torch.requires_grad = True
        u_torch.requires_grad = True
        xdot = self.forward(x_torch, u_torch)
        A = np.empty((12, 12))
        B = np.empty((12, 4))
        for i in range(12):
            if x_torch.grad is not None:
                x_torch.grad.zero_()
            if u_torch.grad is not None:
                u_torch.grad.zero_()
            xdot[0, i].backward(retain_graph=True)
            A[i, :] = x_torch.grad[0, :].detach().numpy()
            B[i, :] = u_torch.grad[0, :].detach().numpy()
        return A, B

    def lqr_control(self, Q, R, x, u):
        """
        The control action should be u = K * (x - x*) + u*
        """
        x_np = x if isinstance(x, np.ndarray) else x.detach().numpy()
        u_np = u if isinstance(u, np.ndarray) else u.detach().numpy()
        A, B = self.linearized_dynamics(x_np, u_np)
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = -np.linalg.solve(R, B.T @ S)
        return K, S

    def _apply(self, fn):
        """Handles CPU/GPU transfer and type conversion."""
        super()._apply(fn)
        self.inertia = fn(self.inertia)
        return self


class QuadraticLyapunov(nn.Module):
    """
    Simple quadratic Lyapunov function.
    """

    def __init__(self, S, **kwargs):
        """
        Args:
          S: the matrix specifying quadratic function x^T S x as the Lyapunov function.
        """
        super().__init__()
        self.register_parameter(name="S", param=torch.nn.Parameter(S.clone().detach()))

    def forward(self, x):
        return torch.sum(x * (x @ self.S), axis=1, keepdim=True)


def create_model(
    dynamics,
    controller_parameters=None,
    lyapunov_parameters=None,
    loss_parameters=None,
    path=None,
    lyapunov_func="lyapunov.NeuralNetworkLyapunov",
    loss_func="lyapunov.LyapunovDerivativeLoss",
    controller_func="controllers.NeuralNetworkController",
):
    """
    Build the computational graph for verification of general dynamics + controller + neural lyapunov function.
    """
    # Default parameters.
    if controller_parameters is None:
        controller_parameters = {
            "nlayer": 2,
            "hidden_dim": 5,
            "clip_output": None,
        }
    if lyapunov_parameters is None:
        lyapunov_parameters = {
            # 'nlayer': 3,
            "hidden_widths": [32, 32],
            "R_rows": 0,
            "absolute_output": False,
            "eps": 0.0,
            "activation": nn.ReLU,
        }
    if loss_parameters is None:
        loss_parameters = {
            "kappa": 0.1,
        }
    controller = eval(controller_func)(
        in_dim=dynamics.x_equilibrium.size(0),
        out_dim=dynamics.u_equilibrium.size(0),
        x_equilibrium=dynamics.x_equilibrium,
        u_equilibrium=dynamics.u_equilibrium,
        **controller_parameters,
    )
    lyapunov_nn = eval(lyapunov_func)(
        x_dim=dynamics.x_equilibrium.size(0),
        goal_state=dynamics.x_equilibrium,
        **lyapunov_parameters,
    )

    loss = eval(loss_func)(dynamics, controller, lyapunov_nn, **loss_parameters)
    # TODO: load a trained model. Currently using random weights, just to test autoLiRPA works.
    if path is not None:
        loss.load_state_dict(torch.load(path))
    return loss


def create_output_feedback_model(
    dynamics,
    controller_parameters,
    lyapunov_parameters,
    path=None,
    observer_parameters=None,
    loss_parameters=None,
    lyapunov_func="lyapunov.NeuralNetworkLyapunov",
    loss_func="lyapunov.DissipativityDerivativeLoss",
    controller_func="controllers.NeuralNetworkController",
    observer_func="controllers.NeuralNetworkLuenbergerObserver",
):
    """
    Build the computational graph for verification of general dynamics + controller + neural lyapunov function.
    """
    if loss_parameters is None:
        loss_parameters = {
            "kappa": 0,
        }
    nx = dynamics.continuous_time_system.nx
    ny = dynamics.continuous_time_system.ny
    nu = dynamics.continuous_time_system.nu
    h = lambda x: dynamics.continuous_time_system.h(x)
    
    # Ensure equilibrium points are on the correct device/dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 # Assuming float32
    x_eq = dynamics.x_equilibrium.to(device).to(dtype)
    u_eq = dynamics.u_equilibrium.to(device).to(dtype)
    
    controller = eval(controller_func)(
        in_dim=nx + ny,
        out_dim=nu,
        x_equilibrium=torch.concat((x_eq, torch.zeros(ny, device=device, dtype=dtype))),
        u_equilibrium=u_eq,
        **controller_parameters,
    )
    lyapunov_nn = eval(lyapunov_func)(
        x_dim=2 * nx,
        goal_state=torch.concat((x_eq, torch.zeros(nx, device=device, dtype=dtype))),
        **lyapunov_parameters,
    )
    observer = eval(observer_func)(
        nx, ny, dynamics, h, torch.zeros(1, ny, device=device, dtype=dtype), observer_parameters["fc_hidden_dim"]
    )
    loss = eval(loss_func)(
        dynamics, observer, controller, lyapunov_nn, **loss_parameters
    )
    if path is not None:
        loss.load_state_dict(torch.load(path)["state_dict"])
    return loss


def create_pendulum_model_state_feedback(**kwargs):
    return create_model(
        dynamical_system.SecondOrderDiscreteTimeSystem(
            pendulum.PendulumDynamics(m=0.15, l=0.5, beta=0.1),
            dt=0.05,
            position_integration=dynamical_system.IntegrationMethod.ExplicitEuler,
            velocity_integration=dynamical_system.IntegrationMethod.ExplicitEuler,
        ),
        **kwargs,
    )


def create_pendulum_model(dt=0.01, **kwargs):
    """
    Build the computational graph for verification of the inverted pendulum model.
    """
    # Create the "model" (the entire computational graph for computing Lyapunov loss). Make sure all parameters here match colab.
    return create_model(
        dynamical_system.SecondOrderDiscreteTimeSystem(
            pendulum.PendulumDynamics(m=0.15, l=0.5, beta=0.1), dt
        ),
        **kwargs,
    )


def create_pendulum_output_feedback_model(dt=0.01, **kwargs):
    """
    Build the computational graph for verification of the inverted pendulum model.
    """
    # Create the "model" (the entire computational graph for computing Lyapunov loss). Make sure all parameters here match colab.
    return create_output_feedback_model(
        dynamical_system.SecondOrderDiscreteTimeSystem(
            pendulum.PendulumDynamics(m=0.15, l=0.5, beta=0.1), dt
        ),
        **kwargs,
    )


def create_pendulum_l2gain_verification_model(
    dt=0.01,
    gamma=1.0,
    w_max=0.05,
    controller_parameters=None,
    lyapunov_parameters=None,
    observer_parameters=None,
    path=None,
):
    """
    Create pendulum output feedback model wrapped for L2-gain verification.
    
    Input to verifier: [xe, w] where xe = [x, e] (state + estimation error)
    Verifies: V(xe) - V(xe_next) + γ²‖w‖² - ‖z‖² ≥ 0 for all (xe, w)
    
    Args:
        dt: Time step
        gamma: L2-gain bound
        w_max: Disturbance bound (scalar)
        controller_parameters: Controller config
        lyapunov_parameters: Lyapunov function config
        observer_parameters: Observer config
        path: Path to pretrained weights
        
    Returns:
        DissipativityVerificationWrapper with input dim = 2*nx + nw = 5
    """
    # Create dynamics
    pendulum_continuous = pendulum.PendulumDynamics(m=0.15, l=0.5, beta=0.1)
    dynamics = dynamical_system.SecondOrderDiscreteTimeSystem(pendulum_continuous, dt)
    
    nx = pendulum_continuous.nx  # 2
    ny = pendulum_continuous.ny  # 1
    nu = pendulum_continuous.nu  # 1
    nw = pendulum_continuous.nw  # 1
    
    # Default parameters matching pendulum_output_training.py
    if controller_parameters is None:
        controller_parameters = {
            "nlayer": 4,
            "hidden_dim": 8,
            "clip_output": "clamp",
            "u_lo": torch.tensor([-0.25]),
            "u_up": torch.tensor([0.25]),
        }
    if lyapunov_parameters is None:
        lyapunov_parameters = {"R_rows": 4, "eps": 0.01}
    if observer_parameters is None:
        observer_parameters = {"fc_hidden_dim": [8, 8]}
    
    # Create controller (input: [z, ey], output: u)
    h = lambda x: pendulum_continuous.h(x)
    controller = controllers.NeuralNetworkController(
        in_dim=nx + ny,
        out_dim=nu,
        x_equilibrium=torch.cat((dynamics.x_equilibrium, torch.zeros(ny))),
        u_equilibrium=dynamics.u_equilibrium,
        **controller_parameters,
    )
    
    # Create Lyapunov function (input: xe, dim = 2*nx)
    lyapunov_nn = lyapunov.NeuralNetworkQuadraticLyapunov(
        goal_state=torch.cat((dynamics.x_equilibrium, torch.zeros(nx))),
        x_dim=2 * nx,
        **lyapunov_parameters,
    )
    
    # Create observer
    observer = controllers.NeuralNetworkLuenbergerObserver(
        nx, ny, dynamics, h, torch.zeros(1, ny), observer_parameters["fc_hidden_dim"]
    )
    
    # Create L2-gain supply rate
    supply_rate_fn = supply_rate.L2GainSupplyRate(gamma=gamma)
    
    # Create dissipativity loss
    loss_fn = lyapunov.DissipativityDerivativeLoss(
        dynamics=dynamics,
        controller=controller,
        lyap_nn=lyapunov_nn,
        supply_rate=supply_rate_fn,
        box_lo=torch.zeros(2*nx, device=dynamics.device), # Dummy
        box_up=torch.zeros(2*nx, device=dynamics.device), # Dummy
        rho_multiplier=1e6, # Ignore ROA
        w_max=torch.tensor([w_max]),
        beta=100,
        hard_max=True,
        loss_weights=torch.tensor([0.0, 1.0, 0.0]), # Only dissipativity
        observer=observer,
    )
    
    # Load pretrained weights if provided
    if path is not None:
        loss_fn.load_state_dict(torch.load(path)["state_dict"])
    
    # Wrap for verification: input is [xe, w], dim = 2*nx + nw = 5
    wrapped_model = lyapunov.DissipativityVerificationWrapper(
        loss_fn=loss_fn,
        state_dim=2 * nx,  # 4
        w_dim=nw,          # 1
    )
    
    return wrapped_model


def create_pendulum_rinn_state_feedback_model(
    dt=0.01,
    gamma=100.0,
    w_max=0.024525,
    rho=0.069,
    s_scale=1.0,
    rinn_parameters=None,
    lyapunov_parameters=None,
    output_C=None,
    verification_mode='combined',
    uncertainty_type='l2gain',
    disk_alpha=0.353,
    disk_sigma=0.0,
    disk_c_bar=3.0,
    kappa=0.0,
    iqc_lambda=1.0,
    iqc_M=None,
):
    """
    Create pendulum state-feedback verification model with RINN controller.

    Builds the full augmented dynamics [x_p, x_k] with dissipativity
    verification.  ρ is baked into the model's min/max condition, so the
    VNNLIB spec only needs to assert output ≥ 0.

    The verifier input is [ξ, w̃] where ξ = [x_p, x_k] (4D) and w̃ is
    the disturbance / free parameter (1D), totalling 5D.

    Supports two uncertainty modes:
      - 'l2gain' (default): Box disturbance w ∈ [−w_max, w_max] with
        L2-gain supply rate s = ‖w‖² − (1/γ²)‖z‖².
      - 'disk_margin': Disk margin D(α, σ) uncertainty via
        DiskMarginTransform.  Supply rate is Lyapunov (s = −κV).
        Verifies: ∀w̃ ∈ [−c̄, c̄], V(ξ⁺(w(ũ, w̃))) ≤ (1−κ)V(ξ).

    Weights are loaded by α,β-CROWN via the ``model.path`` config field.
    """
    # -- plant dynamics --
    pend_ct = pendulum.PendulumDynamics(
        m=0.15, l=0.5, beta=0.1,
        output_C=torch.tensor(output_C) if output_C is not None else None,
    )
    plant = dynamical_system.SecondOrderDiscreteTimeSystem(
        pend_ct, dt,
        position_integration=dynamical_system.IntegrationMethod.MidPoint,
        velocity_integration=dynamical_system.IntegrationMethod.ExplicitEuler,
    )

    n_plant = plant.nx  # 2

    # -- RINN controller --
    rp = rinn_parameters or {}
    _t = lambda v: torch.tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v

    rinn_ctrl = controllers.RINNController(
        A=_t(rp['A']),   Bw=_t(rp['Bw']),  By=_t(rp['By']),
        Cv=_t(rp['Cv']),  Dvw=_t(rp['Dvw']), Dvy=_t(rp['Dvy']),
        Cu=_t(rp['Cu']),  Duw=_t(rp['Duw']), Duy=_t(rp['Duy']),
        n_plant=n_plant, dt=dt,
        output_fn=None,
        trainable=True,
        freeze_dvw_lower_tri=True,
        activation=rp.get('activation', 'relu'),
        clip_output='clamp' if 'u_lo' in rp else None,
        u_lo=_t(rp['u_lo']) if 'u_lo' in rp else None,
        u_up=_t(rp['u_up']) if 'u_up' in rp else None,
    )

    # -- augmented dynamics --
    aug_dyn = dynamical_system.AugmentedRINNDynamics(plant, rinn_ctrl)
    nx = aug_dyn.nx

    # -- Lyapunov function --
    lp = dict(lyapunov_parameters or {})
    if 'R_trainable' not in lp:
        lp['R_trainable'] = torch.zeros(nx, nx)
    if 'R_frozen' not in lp:
        lp['R_frozen'] = torch.zeros(nx, nx)
    lyap_nn = lyapunov.NeuralNetworkLyapunov(
        goal_state=aug_dyn.x_equilibrium,
        x_dim=nx,
        **lp,
    )

    # -- supply rate & uncertainty --
    from neural_lyapunov_training.uncertainty import (
        DiskMarginTransform, SectorBoundTransform,
        disk_margin_iqc_z_fn, disk_margin_iqc_M,
    )

    uncertainty_xform = None
    iqc_kwargs = {}

    if uncertainty_type == 'iqc':
        supply_fn = supply_rate.LyapunovSupplyRate(kappa=kappa)
        effective_w_max = torch.tensor([w_max])
        # IQC multiplier for CROWN verification
        M_tensor = torch.tensor(iqc_M, dtype=torch.float32) if iqc_M is not None else disk_margin_iqc_M(disk_alpha)
        z_fn = disk_margin_iqc_z_fn(sigma=disk_sigma)
        iqc_kwargs = {
            'iqc_M': M_tensor,
            'iqc_z_fn': z_fn,
            'iqc_lambda_init': iqc_lambda,
            'learnable_iqc_lambda': False,
        }
    elif uncertainty_type == 'sector_bound':
        supply_fn = supply_rate.LyapunovSupplyRate(kappa=kappa)
        uncertainty_xform = SectorBoundTransform(alpha=disk_alpha)
        effective_w_max = torch.tensor([1.0])
    elif uncertainty_type == 'disk_margin':
        supply_fn = supply_rate.LyapunovSupplyRate(kappa=kappa)
        uncertainty_xform = DiskMarginTransform(
            alpha=disk_alpha, sigma=disk_sigma, c_bar=disk_c_bar,
        )
        effective_w_max = torch.tensor([disk_c_bar])
    else:
        supply_fn = supply_rate.L2GainSupplyRate(gamma=gamma)
        effective_w_max = torch.tensor([w_max])

    loss_fn = lyapunov.DissipativityDerivativeLoss(
        dynamics=aug_dyn,
        controller=rinn_ctrl,
        lyap_nn=lyap_nn,
        supply_rate=supply_fn,
        box_lo=torch.zeros(nx),
        box_up=torch.zeros(nx),
        rho_multiplier=1.0,
        w_max=effective_w_max,
        beta=100,
        hard_max=True,
        s_scale=s_scale,
        relu_minmax=True,
        verification_mode=verification_mode,
        uncertainty_transform=uncertainty_xform,
        **iqc_kwargs,
    )

    loss_fn._fixed_rho = rho
    return loss_fn


def create_flexible_rod_rinn_model(
    dt=0.005,
    rho=0.0,
    s_scale=1.0,
    gamma_delta=0.1,
    c_bar=3.0,
    n_k=2,
    n_w_rinn=8,
    activation='relu',
    u_max=20.0,
    controller_limit=None,
    m_b=1.0,
    m_t=0.1,
    L=1.0,
    rho_l=0.1,
    lyapunov_parameters=None,
    output_C=None,
    verification_mode='combined',
    kappa=0.0,
    nominal=False,
    supply_type='lyapunov',
    gamma=100.0,
    w_max=0.0,
):
    """
    Create flexible rod + RINN + tanh norm-ball verification model.

    Builds the full augmented dynamics [x_p, x_k] with Lyapunov stability
    verification and tanh norm-ball uncertainty parameterization.

    The verifier input is [ξ, w̃] where ξ = [x_p, x_k] (4D) and w̃ is
    the free uncertainty parameter (1D), totalling 5D.

    Internally, w̃ is mapped to w via TanhNormBallTransform:
        w = γ_Δ ||u|| tanh(||w̃||) / (||w̃|| + ε) · w̃
    guaranteeing ||w|| < γ_Δ ||u|| for all w̃.

    Weights are loaded by α,β-CROWN via the ``model.path`` config field.
    Dummy matrices are used for RINN init; CROWN overwrites from checkpoint.

    Args:
        dt: Discretisation time step.
        rho: Fixed ROA sublevel set value (for bisection).
        s_scale: Supply rate scaling factor.
        gamma_delta: Uncertainty gain bound ||Δ|| ≤ γ_Δ.
        c_bar: Free parameter box bound: w̃ ∈ [-c̄, c̄].
        n_k: RINN controller state dimension.
        n_w_rinn: RINN implicit-layer (hidden) dimension.
        activation: RINN activation function ('relu' or 'leaky_relu').
        u_max: Control saturation bound.
        controller_limit: Controller state limits (list of n_k values).
        m_b, m_t, L, rho_l: Flexible rod physical parameters.
        lyapunov_parameters: Dict with Lyapunov function config.
        output_C: Performance output matrix z = C_z * x_p.
        verification_mode: 'combined', 'dissipativity', or 'invariance'.
        kappa: Lyapunov decay rate (0 = simple stability).
        supply_type: 'lyapunov' or 'l2gain'. L2-gain adds ||w||²-(1/γ²)||z||²
            slack that makes CROWN verification tractable.
        gamma: L2-gain bound (only used when supply_type='l2gain').
        w_max: Box disturbance bound (only used when supply_type='l2gain'
            and nominal=True for non-uncertainty L2-gain verification).
    """
    import neural_lyapunov_training.flexible_rod as flexible_rod_mod
    from neural_lyapunov_training.uncertainty import TanhNormBallTransform
    from neural_lyapunov_training.systems import LinearMeasurement

    # -- plant dynamics (CT → DT via Euler) --
    rod_ct = flexible_rod_mod.FlexibleRodDynamics(
        m_b=m_b, m_t=m_t, L=L, rho_l=rho_l,
        output_C=torch.tensor(output_C) if output_C is not None else None,
    )
    plant = dynamical_system.FirstOrderDiscreteTimeSystem(
        rod_ct, dt=dt,
        integration=dynamical_system.IntegrationMethod.ExplicitEuler,
    )

    n_plant = plant.nx  # 2
    n_y = rod_ct.ny     # 1 (position measurement)
    n_u = rod_ct.nu     # 1

    # -- RINN controller (dummy matrices — CROWN overwrites from checkpoint) --
    # output_fn: y = Cpy @ x_p (position-only measurement)
    measurement_fn = LinearMeasurement(torch.zeros(n_y, n_plant))

    rinn_ctrl = controllers.RINNController(
        A=torch.zeros(n_k, n_k),
        Bw=torch.zeros(n_k, n_w_rinn),
        By=torch.zeros(n_k, n_y),
        Cv=torch.zeros(n_w_rinn, n_k),
        Dvw=torch.zeros(n_w_rinn, n_w_rinn),
        Dvy=torch.zeros(n_w_rinn, n_y),
        Cu=torch.zeros(n_u, n_k),
        Duw=torch.zeros(n_u, n_w_rinn),
        Duy=torch.zeros(n_u, n_y),
        n_plant=n_plant,
        dt=dt,
        output_fn=measurement_fn,
        trainable=True,
        freeze_dvw_lower_tri=True,
        activation=activation,
        clip_output='clamp',
        u_lo=torch.tensor([-u_max]),
        u_up=torch.tensor([u_max]),
    )

    # -- augmented dynamics [x_p, x_k] --
    aug_dyn = dynamical_system.AugmentedRINNDynamics(plant, rinn_ctrl)
    nx = aug_dyn.nx  # n_plant + n_k

    # -- Lyapunov function --
    lp = dict(lyapunov_parameters or {})
    if 'R_trainable' not in lp:
        lp['R_trainable'] = torch.zeros(nx, nx)
    if 'R_frozen' not in lp:
        lp['R_frozen'] = torch.zeros(nx, nx)
    lyap_nn = lyapunov.NeuralNetworkLyapunov(
        goal_state=aug_dyn.x_equilibrium,
        x_dim=nx,
        **lp,
    )

    # -- Supply rate --
    if supply_type == 'l2gain':
        supply_fn = supply_rate.L2GainSupplyRate(gamma=gamma)
    else:
        supply_fn = supply_rate.LyapunovSupplyRate(kappa=kappa)

    if nominal:
        # Nominal verification: no uncertainty transform
        # For L2-gain, still need w_max for the disturbance dimension
        if supply_type == 'l2gain' and w_max > 0:
            effective_w_max = torch.tensor([w_max])
        else:
            effective_w_max = torch.tensor([0.0])
        loss_fn = lyapunov.DissipativityDerivativeLoss(
            dynamics=aug_dyn,
            controller=rinn_ctrl,
            lyap_nn=lyap_nn,
            supply_rate=supply_fn,
            box_lo=torch.zeros(nx),
            box_up=torch.zeros(nx),
            rho_multiplier=1.0,
            w_max=effective_w_max,
            beta=100,
            hard_max=True,
            s_scale=s_scale,
            relu_minmax=True,
            verification_mode=verification_mode,
            uncertainty_transform=None,
        )
    else:
        # -- Tanh norm-ball uncertainty transform: w̃ → w --
        uncertainty_xform = TanhNormBallTransform(
            gamma_delta=gamma_delta,
            n_w=rod_ct.nw,  # 1 (plant uncertainty dimension)
            c_bar=c_bar,
        )

        # -- Dissipativity loss (the verification model) --
        loss_fn = lyapunov.DissipativityDerivativeLoss(
            dynamics=aug_dyn,
            controller=rinn_ctrl,
            lyap_nn=lyap_nn,
            supply_rate=supply_fn,
            box_lo=torch.zeros(nx),
            box_up=torch.zeros(nx),
            rho_multiplier=1.0,
            w_max=torch.tensor([c_bar]),  # bounds on w̃ (for w̃-domain splitting)
            beta=100,
            hard_max=True,
            s_scale=s_scale,
            relu_minmax=True,
            verification_mode=verification_mode,
            uncertainty_transform=uncertainty_xform,
        )

    loss_fn._fixed_rho = rho

    return loss_fn


def create_path_tracking_model(dt=0.05, **kwargs):
    """
    Build the computational graph for verification of the inverted pendulum model.
    """
    # Create the "model" (the entire computational graph for computing Lyapunov loss). Make sure all parameters here match colab.
    return create_model(
        dynamical_system.FirstOrderDiscreteTimeSystem(
            path_tracking.PathTrackingDynamics(speed=2.0, length=1.0, radius=10.0), dt
        ),
        **kwargs,
    )


def create_quadrotor2d_model(dt=0.01, **kwargs):
    """
    Build the computational graph for verification of the Quadrotor2D model.
    """
    return create_model(
        dynamical_system.SecondOrderDiscreteTimeSystem(
            quadrotor2d.Quadrotor2DDynamics(),
            dt=dt,
        ),
        # pretrained_path='lyaloss_quadroter2d.pth',
        **kwargs,
    )


def create_quadrotor2d_output_feedback_model(dt=0.01, **kwargs):
    """
    Build the computational graph for verification of the Quadrotor2D model.
    """
    return create_output_feedback_model(
        dynamical_system.SecondOrderDiscreteTimeSystem(
            quadrotor2d.Quadrotor2DLidarDynamics(
                length=0.25, mass=0.486, inertia=0.00383, gravity=9.81
            ),
            dt=dt,
        ),
        **kwargs,
    )


def create_continuous_time_quadrotor2d_model(**kwargs):
    """
    Build the computational graph for verification of the Quadrotor2D model.
    """
    return create_model(
        quadrotor2d.Quadrotor2DDynamics(
            length=0.25, mass=0.486, inertia=0.00383, gravity=9.81
        ),
        # pretrained_path='lyaloss_quadroter2d.pth',
        **kwargs,
    )


def create_pvtol_model(**kwargs):
    """
    Build the computational graph for verification of the Pvtol model.
    """
    return create_model(pvtol.PvtolDynamics(), **kwargs)


def create_quadrotor3d_model(**kwargs):
    """
    Build the computational graph for verification of the Quadrotor3D model.
    """
    return create_model(
        Quadrotor3DDynamics(dtype=torch.float32),
        # pretrained_path='lyaloss_quadroter2d.pth',
        **kwargs,
    )


def create_cartpole_model(**kwargs):
    """
    Build the computational graph for verification of the Acrobot model.
    """
    return create_model(
        CartPoleDynamics(),
        controller_parameters={"nlayer": 1},
        # pretrained_path='lyaloss_acrobot.pth',
        **kwargs,
    )


def create_acrobot_model(**kwargs):
    """
    Build the computational graph for verification of the Acrobot model.
    """
    return create_model(
        AcrobotDynamics(),
        controller_parameters={"nlayer": 1},
        # pretrained_path='lyaloss_acrobot.pth',
        **kwargs,
    )


def add_hole(box_low, box_high, inner_low, inner_high):
    boxes_low = []
    boxes_high = []
    for i in range(box_low.size(0)):
        # Split on dimension i.
        box1_low = box_low.clone()
        box1_low[i] = inner_high[i]
        box1_high = box_high.clone()
        box2_low = box_low.clone()
        box2_high = box_high.clone()
        box2_high[i] = inner_low[i]
        boxes_low.extend([box1_low, box2_low])
        boxes_high.extend([box1_high, box2_high])
        box_low[i] = inner_low[i]
        box_high[i] = inner_high[i]
    boxes_low = torch.stack(boxes_low, dim=0)
    boxes_high = torch.stack(boxes_high, dim=0)
    return boxes_low, boxes_high


def box_data(
    eps=None, lower_limit=-1.0, upper_limit=1.0, ndim=2, scale=1.0, hole_size=0
):
    """
    Generate a box between (-1, -1) and (1, 1) as our region to verify stability.
    We may place a small hole around the origin.
    """
    if isinstance(lower_limit, list):
        data_min = scale * torch.tensor(
            lower_limit, dtype=torch.get_default_dtype()
        ).unsqueeze(0)
    else:
        data_min = scale * torch.ones((1, ndim)) * lower_limit
    if isinstance(upper_limit, list):
        data_max = scale * torch.tensor(
            upper_limit, dtype=torch.get_default_dtype()
        ).unsqueeze(0)
    else:
        data_max = scale * torch.ones((1, ndim)) * upper_limit
    if hole_size != 0:
        inner_low = data_min.squeeze(0) * hole_size
        inner_high = data_max.squeeze(0) * hole_size
        data_min, data_max = add_hole(
            data_min.squeeze(0), data_max.squeeze(0), inner_low, inner_high
        )
    X = (data_min + data_max) / 2.0
    # Assume the "label" is 1, so we verify the positiveness.
    labels = torch.ones(size=(data_min.size(0),), dtype=torch.int64)
    # Lp norm perturbation epsilon. Not used, since we will return per-element min and max.
    eps = None
    return X, labels, data_max, data_min, eps


def simulate(lyaloss, steps: int, x0):
    # Assumes explicit euler integration.
    x_traj = [None] * steps
    V_traj = [None] * steps
    x_traj[0] = x0
    
    with torch.no_grad():
        if hasattr(lyaloss, 'observer') and lyaloss.observer is not None:
            # Output feedback simulation
            nx = lyaloss.nx
            device = x0.device
            dtype = x0.dtype
            
            # Start estimate z at equilibrium
            try:
                z = lyaloss.dynamics.continuous_time_system.x_equilibrium.to(device).to(dtype).repeat(x0.shape[0], 1)
            except:
                z = torch.zeros((x0.shape[0], nx), device=device, dtype=dtype)
                
            current_x = x0
            current_z = z
            xe = torch.cat((current_x, current_x - current_z), dim=1)
            V_traj[0] = lyaloss.lyapunov.forward(xe)
            
            for i in range(1, steps):
                y = lyaloss.observer.h(current_x)
                ey = y - lyaloss.observer.h(current_z)
                u = lyaloss.controller.forward(torch.cat((current_z, ey), dim=1))
                
                next_x = lyaloss.dynamics.forward(current_x, u)
                next_z = lyaloss.observer.forward(current_z, u, y)
                
                current_x = next_x
                current_z = next_z
                xe = torch.cat((current_x, current_x - current_z), dim=1)
                
                x_traj[i] = current_x
                V_traj[i] = lyaloss.lyapunov.forward(xe)
        else:
            # State feedback simulation
            V_traj[0] = lyaloss.lyapunov.forward(x_traj[0])
            for i in range(1, steps):
                u = lyaloss.controller.forward(x_traj[i - 1])
                x_traj[i] = lyaloss.dynamics.forward(x_traj[i - 1], u)
                V_traj[i] = lyaloss.lyapunov.forward(x_traj[i])

    return x_traj, V_traj
