# Verify local dissipativity of a controller using QCs


import time
from typing import Literal, assert_never

import cvxpy as cp
import numpy as np
import torch
from joblib import Parallel, delayed

from qc_utilities import (
    ControllerThetaParameters,
    PlantParameters,
    construct_closed_loop,
    matprint,
)


def flexible_rod_setup(
    alpha_rel_normP=1.0, alpha_absolute=None, saturation_absolute=20.0
):
    dt = 0.005
    # dt = 0.001
    mb = 1  # Mass of base (Kg)
    mt = 0.1  # Mass of tip (Kg)
    L = 1  # Length of link (m)
    rho = 0.1  # Mass density of Link (N/m)

    # r = 1e-2  # Radius of rod cross-section (m)
    # E = 200e9  # Young's modulus for steel, GPa
    # I = (np.pi / 4) * r**4  # Area second moment of inertia (m^4)
    # EI = E * I
    # flexdamp = 0.9
    Mr = mb + mt + rho * L  # Total mass (Kg)
    Ap = np.array([[0, 1], [0, 0]], dtype=np.float32)
    Bpw = np.array([[1], [0]], dtype=np.float32)
    Bpu = np.array([[0], [1.0 / Mr]], dtype=np.float32)
    Bpd = np.zeros(Bpu.shape)
    Cpv = np.zeros((1, 2), dtype=np.float32)
    Dpvw = np.zeros((1, 1), dtype=np.float32)
    Dpvu = np.eye(1, dtype=np.float32)
    Dpvd = np.zeros(Dpvu.shape)
    Cpe = np.eye(2, dtype=np.float32)
    Dpew = np.zeros((2, 1), dtype=np.float32)
    Dped = np.zeros((2, 1), dtype=np.float32)
    Dpeu = np.zeros((2, 1), dtype=np.float32)
    Cpy = np.array([[1, 0]], dtype=np.float32)
    Dpyw = np.zeros((1, 1), dtype=np.float32)
    Dpyd = np.zeros((1, 1), dtype=np.float32)

    weights = torch.load("controllers/flexible_rod_rinn8.pth", weights_only=False)
    k0 = ControllerThetaParameters(
        Ak=weights["A_T"].T.numpy(),
        Bkw=weights["Bw_T"].T.numpy(),
        Bky=weights["By_T"].T.numpy(),
        Ckv=weights["Cv_T"].T.numpy(),
        Dkvw=weights["Dvw_T"].T.numpy(),
        Dkvy=weights["Dvy_T"].T.numpy(),
        Cku=weights["Cu_T"].T.numpy(),
        Dkuw=weights["Duw_T"].T.numpy(),
        Dkuy=weights["Duy_T"].T.numpy(),
    )
    P = weights["P"].numpy()
    nphi = k0.Dkvw.shape[0]
    k = k0.wrap_in_saturation(saturation_absolute)

    normP = np.linalg.norm(P)
    P = P / normP
    if alpha_absolute:
        alpha = alpha_absolute
    else:
        alpha = alpha_rel_normP / normP

    def controller_nonlin_constraints(eps=0.0, bound_region=1.0):
        # saturation constraint is valid in [-bound_region, bound_region]
        assert bound_region >= 1.0
        nu = 1
        Lambda = cp.Variable((nphi, nphi), diag=True)
        lsat = cp.Variable(nonneg=True)
        MDeltavv = cp.bmat(
            [
                [np.zeros((nphi, nphi + nu))],
                [np.zeros((nu, nphi)), np.array([[-2]]) * lsat],
            ]
        )
        MDeltavw = cp.bmat(
            [
                [Lambda, np.zeros((nphi, nu))],
                [np.zeros((nu, nphi)), np.array([[1 + bound_region]]) * lsat],
            ]
        )
        MDeltaww = cp.bmat(
            [
                [-2 * Lambda, np.zeros((nphi, nu))],
                [np.zeros((nu, nphi)), np.array([[-2 * bound_region]]) * lsat],
            ]
        )
        constraints = [Lambda >> 0]
        variables = [Lambda, lsat]
        return (MDeltavv, MDeltavw, MDeltaww, variables, constraints)

    def plant_uncertainty_constraints(eps=0.0, bound_region=0.0):
        # Ignores bound region
        lagrange_multiplier = 5.0
        delta_alpha = 1.0
        # b = 0.1 # The real one
        b = 0.09
        MDeltapvv_fixed = (
            lagrange_multiplier * b**2 * delta_alpha * np.array([[1]], dtype=np.float32)
        )
        MDeltapvw_fixed = np.array([[0]], dtype=np.float32)
        MDeltapww_fixed = lagrange_multiplier * np.array([[-1]], dtype=np.float32)
        Lambda = cp.Variable(nonneg=True)
        MDeltapvv = Lambda * MDeltapvv_fixed
        MDeltapvw = Lambda * MDeltapvw_fixed
        MDeltapww = Lambda * MDeltapww_fixed
        variables = [Lambda]
        constraints = []
        return (MDeltapvv, MDeltapvw, MDeltapww, variables, constraints)

    def closed_loop_cstor(
        MDeltapvv, MDeltapvw, MDeltapww, MDeltakvv, MDeltakvw, MDeltakww, stacker
    ):
        # MDeltap is QC for plant uncertainty
        # Lambda is parameter of QC for controller nonlinearity
        controller_params = ControllerThetaParameters(
            k.Ak,
            k.Bkw,
            k.Bky,
            k.Ckv,
            k.Dkvw,
            k.Dkvy,
            k.Cku,
            k.Dkuw,
            k.Dkuy,
            MDeltakvv=MDeltakvv,
            MDeltakvw=MDeltakvw,
            MDeltakww=MDeltakww,
        )
        plant_params = PlantParameters(
            Ap,
            Bpw,
            Bpd,
            Bpu,
            Cpv,
            Dpvw,
            Dpvd,
            Dpvu,
            Cpe,
            Dpew,
            Dped,
            Dpeu,
            Cpy,
            Dpyw,
            Dpyd,
            MDeltapvv,
            MDeltapvw,
            MDeltapww,
        )
        # LDeltap = MDeltavv_to_LDelta(plant_params.MDeltapvv)

        A, Bw, Bd, Cv, Dvw, Dvd, Ce, Dew, Ded, Mvv, Mvw, Mww = construct_closed_loop(
            plant_params, controller_params, stacker, LDeltap=None
        )

        # Convert to discrete time with Euler integration
        Ad = np.eye(A.shape[0]) + A * dt
        Bwd = Bw * dt
        Bdd = Bd * dt
        Cvd, Dvwd, Dvdd, Ced, Dewd, Dedd = Cv, Dvw, Dvd, Ce, Dew, Ded

        return (
            Ad,
            Bwd,
            Bdd,
            Cvd,
            Dvwd,
            Dvdd,
            Ced,
            Dewd,
            Dedd,
            Mvv,
            Mvw,
            Mww,
            P,
            alpha,
            normP,
            plant_params,
            controller_params,
        )

    return (
        P,
        controller_nonlin_constraints,
        plant_uncertainty_constraints,
        closed_loop_cstor,
    )


def pendulum_setup(
    type: Literal["disturbance", "model_uncertainty"],
    alpha_rel_normP=1.0,
    alpha_absolute=None,
    saturation_rel_mgl=0.333,
    uncertainty_bound=0.5,
):
    dt = 0.01
    g = 9.8
    m = 0.15
    l = 0.5
    # mu = 0.05
    mu = 0.1

    # new hidden size 8 rinn
    Ak0 = np.array([[-3.1509, -1.4492], [-1.1005, -0.6330]])
    Bkw0 = np.array(
        [
            [0.0236, -0.0235, -0.0235, 0.0245, 0.0232, -0.0232, 0.0236, -0.0231],
            [0.0219, -0.0219, -0.0218, 0.0229, 0.0215, -0.0215, 0.0221, -0.0214],
        ]
    )
    Bky0 = np.array([[-0.0151, 0.0151], [-0.0204, 0.0978]])
    Ckv0 = np.array(
        [
            [0.0188, -0.0085],
            [-0.0185, 0.0083],
            [-0.0188, 0.0085],
            [0.0250, -0.0127],
            [0.0210, -0.0100],
            [-0.0193, 0.0090],
            [0.0228, -0.0111],
            [-0.0192, 0.0090],
        ]
    )
    Dkvw0 = np.array(
        [
            [0.0000, -0.0327, -0.0327, 0.0348, 0.0327, -0.0323, 0.0332, -0.0320],
            [0.0000, 0.0000, 0.0327, -0.0348, -0.0327, 0.0323, -0.0332, 0.0320],
            [0.0000, 0.0000, 0.0000, -0.0348, -0.0327, 0.0323, -0.0332, 0.0320],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0313, -0.0304, 0.0322, -0.0302],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0319, 0.0331, -0.0316],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0333, 0.0321],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0309],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        ]
    )
    Dkvy0 = np.array(
        [
            [-0.0063, 0.0235],
            [0.0065, -0.0234],
            [0.0063, -0.0235],
            [-0.0072, 0.0275],
            [-0.0059, 0.0248],
            [0.0057, -0.0238],
            [-0.0066, 0.0259],
            [0.0056, -0.0238],
        ]
    )
    Cku0 = np.array([[0.0906, 0.0395]])
    Dkuw0 = np.array(
        [[0.0342, -0.0338, -0.0340, 0.0353, 0.0343, -0.0346, 0.0338, -0.0344]]
    )
    Dkuy0 = np.array([[-2.7673, -2.1278]])
    P = np.array(
        [
            [1.5211e-02, 1.1826e-03, -3.9104e-05, 7.0204e-05],
            [1.1826e-03, 1.1001e-03, 1.4486e-04, 5.6436e-05],
            [-3.9104e-05, 1.4486e-04, 1.3817e-02, 4.1302e-03],
            [7.0204e-05, 5.6436e-05, 4.1302e-03, 3.8687e-03],
        ]
    )

    # Wrap in saturation
    saturation = saturation_rel_mgl * m * g * l
    Ak = Ak0
    Bkw = np.hstack([Bkw0, np.zeros((Ak0.shape[0], 1))])
    Bky = Bky0
    Ckv = np.vstack([Ckv0, Cku0 / saturation])
    Dkvw = np.hstack(
        [
            np.vstack([Dkvw0, Dkuw0 / saturation]),
            np.zeros((Dkvw0.shape[0] + Dkuw0.shape[0], 1)),
        ]
    )
    Dkvy = np.vstack([Dkvy0, Dkuy0 / saturation])
    Cku = np.zeros((1, Ak0.shape[0]))
    Dkuw = np.hstack([np.zeros((1, Dkvw0.shape[0])), np.array([[saturation]])])
    Dkuy = np.zeros((1, Dkuy0.shape[1]))

    normP = np.linalg.norm(P)
    P = P / normP
    if alpha_absolute:
        alpha = alpha_absolute
    else:
        alpha = alpha_rel_normP / normP

    nx = 2
    nu = 1
    ny = 2

    match type:
        case "disturbance":
            nv = 1
            nw = 1
            ne = nx
            nd = nu
            # thetadotdot = (1/ml^2) (-mu thetadot + mgl sintheta + u + d)
            Ap = np.array([[0, 1], [0, -mu / (m * l**2)]], dtype=np.float32)
            Bpw = np.array([[0], [g / l]], dtype=np.float32)
            Bpu = np.array([[0], [1 / (m * l**2)]], dtype=np.float32)
            Bpd = Bpu.copy()
            Cpv = np.array([[1, 0]], dtype=np.float32)
            Dpvw = np.zeros((nv, nw), dtype=np.float32)
            Dpvu = np.zeros((nv, nu), dtype=np.float32)
            Dpvd = Dpvu.copy()
            Cpe = np.eye(nx, dtype=np.float32)
            Dpew = np.zeros((ne, nw), dtype=np.float32)
            Dped = np.zeros((ne, nd), dtype=np.float32)
            Dpeu = np.zeros((ne, nu), dtype=np.float32)
            Cpy = np.eye(nx, dtype=np.float32)
            Dpyw = np.zeros((ny, nw), dtype=np.float32)
            Dpyd = np.zeros((ny, nd), dtype=np.float32)
        case "model_uncertainty":
            # thetadotdot = (1/ml^2) (-mu thetadot + mgl sintheta + u + w2)
            # where alpha^2 u^2 - w2^2 >= 0
            # v1 = x1, v2 = u
            # w1 = sin(x1)
            nv = 2
            nw = 2
            # Setting these to 1 because some of my functions don't like 0 numbers
            ne = 1
            nd = 1
            Ap = np.array([[0, 1], [0, -mu / (m * l**2)]], dtype=np.float32)
            Bpw = np.array([[0.0, 0.0], [g / l, 1 / (m * l**2)]], dtype=np.float32)
            Bpu = np.array([[0.0], [1 / (m * l**2)]], dtype=np.float32)
            Bpd = np.zeros((nx, nd))
            Cpv = np.array([[1.0, 0], [0, 0]])
            Dpvw = np.zeros((nv, nw))
            Dpvu = np.array([[0.0], [1.0]])
            Dpvd = np.zeros((nv, nu))
            Cpe = np.zeros((ne, nx))
            Dpew = np.zeros((ne, nw), dtype=np.float32)
            Dped = np.zeros((ne, nd), dtype=np.float32)
            Dpeu = np.zeros((ne, nu), dtype=np.float32)
            Cpy = np.eye(nx, dtype=np.float32)
            Dpyw = np.zeros((ny, nw), dtype=np.float32)
            Dpyd = np.zeros((ny, nd), dtype=np.float32)
        case _:
            assert_never(type)

    def controller_nonlin_constraints(eps, bound_region=10.0):
        # saturation constraint is valid in [-bound_region, bound_region]
        assert bound_region >= 1.0
        nphi = Dkvw0.shape[0]
        nu = 1
        Lambda = cp.Variable((nphi, nphi), diag=True)
        lsat = cp.Variable(nonneg=True)
        MDeltavv = cp.bmat(
            [
                [np.zeros((nphi, nphi + nu))],
                [np.zeros((nu, nphi)), np.array([[-2]]) * lsat],
            ]
        )
        MDeltavw = cp.bmat(
            [
                [Lambda, np.zeros((nphi, nu))],
                [np.zeros((nu, nphi)), np.array([[1 + bound_region]]) * lsat],
            ]
        )
        MDeltaww = cp.bmat(
            [
                [-2 * Lambda, np.zeros((nphi, nu))],
                [np.zeros((nu, nphi)), np.array([[-2 * bound_region]]) * lsat],
            ]
        )
        constraints = [Lambda >> 0]
        variables = [Lambda, lsat]
        return (MDeltavv, MDeltavw, MDeltaww, variables, constraints)

    def plant_uncertainty_constraints(eps, bound_region=np.pi):
        # Cosntructs constraint on sin valid for [-bound_region, bound_region]
        if bound_region < 4.49341:
            C_sin = np.sin(bound_region) / bound_region
        else:
            # Global bound past this point
            C_sin = np.sin(4.49341) / 4.49341
        D_sin = 1
        lambda_sin = cp.Variable(nonneg=True)
        M1vv = lambda_sin * np.array([[-2 * C_sin * D_sin]], dtype=np.float32)
        M1vw = lambda_sin * np.array([[C_sin + D_sin]], dtype=np.float32)
        M1ww = lambda_sin * np.array([[-2]], dtype=np.float32)

        lambda_uncertainty = cp.Variable(nonneg=True)
        M2vv = lambda_uncertainty * np.array([[uncertainty_bound**2]], dtype=np.float32)
        M2vw = lambda_uncertainty * np.zeros((1, 1))
        M2ww = lambda_uncertainty * np.array([[-1.0]])

        match type:
            case "disturbance":
                MDeltapvv = M1vv
                MDeltapvw = M1vw
                MDeltapww = M1ww
                variables = [lambda_sin]
                constraints = []
            case "model_uncertainty":
                MDeltapvv = cp.bmat(
                    [[M1vv, np.zeros((1, 1))], [np.zeros((1, 1)), M2vv]]
                )
                MDeltapvw = cp.bmat(
                    [[M1vw, np.zeros((1, 1))], [np.zeros((1, 1)), M2vw]]
                )
                MDeltapww = cp.bmat(
                    [[M1ww, np.zeros((1, 1))], [np.zeros((1, 1)), M2ww]]
                )
                variables = [lambda_sin, lambda_uncertainty]
                constraints = []
            case _:
                assert_never(type)

        return (MDeltapvv, MDeltapvw, MDeltapww, variables, constraints)

    def closed_loop_cstor(
        MDeltapvv, MDeltapvw, MDeltapww, MDeltakvv, MDeltakvw, MDeltakww, stacker
    ):
        # MDeltap is QC for plant uncertainty
        # Lambda is parameter of QC for controller nonlinearity
        controller_params = ControllerThetaParameters(
            Ak,
            Bkw,
            Bky,
            Ckv,
            Dkvw,
            Dkvy,
            Cku,
            Dkuw,
            Dkuy,
            MDeltakvv=MDeltakvv,
            MDeltakvw=MDeltakvw,
            MDeltakww=MDeltakww,
        )
        plant_params = PlantParameters(
            Ap,
            Bpw,
            Bpd,
            Bpu,
            Cpv,
            Dpvw,
            Dpvd,
            Dpvu,
            Cpe,
            Dpew,
            Dped,
            Dpeu,
            Cpy,
            Dpyw,
            Dpyd,
            MDeltapvv,
            MDeltapvw,
            MDeltapww,
        )
        # LDeltap = MDeltavv_to_LDelta(plant_params.MDeltapvv)

        A, Bw, Bd, Cv, Dvw, Dvd, Ce, Dew, Ded, Mvv, Mvw, Mww = construct_closed_loop(
            plant_params, controller_params, stacker, LDeltap=None
        )

        # Convert to discrete time with Euler integration
        Ad = np.eye(A.shape[0]) + A * dt
        Bwd = Bw * dt
        Bdd = Bd * dt
        Cvd, Dvwd, Dvdd, Ced, Dewd, Dedd = Cv, Dvw, Dvd, Ce, Dew, Ded

        return (
            Ad,
            Bwd,
            Bdd,
            Cvd,
            Dvwd,
            Dvdd,
            Ced,
            Dewd,
            Dedd,
            Mvv,
            Mvw,
            Mww,
            P,
            alpha,
            normP,
            plant_params,
            controller_params,
        )

    return (
        P,
        controller_nonlin_constraints,
        plant_uncertainty_constraints,
        closed_loop_cstor,
    )


def rfi_fixed_x1_c_dt(
    P,
    ubar,
    sat_bound_region,
    sin_bound_region,
    c,
    dbar,
    controller_nonlin_constraints,
    plant_uncertainty_constraints,
    closed_loop_cstor,
    eps=1e-6,
    quiet=False,
):
    # TODO: does NOT verify whether the level set fits within the area that the saturation bound holds. Have to do that after the fact.
    MDeltakvv, MDeltakvw, MDeltakww, _, MDeltak_constraints = (
        controller_nonlin_constraints(eps, bound_region=sat_bound_region)
    )
    MDeltapvv, MDeltapvw, MDeltapww, _, MDeltap_constraints = (
        plant_uncertainty_constraints(eps, bound_region=sin_bound_region)
    )
    A, Bw, Bd, Cv, Dvw, Dvd, Ce, Dew, Ded, Mvv, Mvw, Mww, _, _, _, p, k = (
        closed_loop_cstor(
            MDeltapvv, MDeltapvw, MDeltapww, MDeltakvv, MDeltakvw, MDeltakww, "cvxpy"
        )
    )

    sc = cp.Variable(nonneg=True)
    sd = cp.Variable(nonneg=True)
    s_sin = cp.Variable(nonneg=True)
    s_sat = cp.Variable(nonneg=True)

    # Core RFI condition: -V(x') + c - sc*(-V(x)+c) - sd*(-d^Td + dbar^2) - z^T M z >= 0
    ABwBd = cp.hstack([A, Bw, Bd])
    nVFxwd = -ABwBd.T @ P @ ABwBd
    x00 = cp.hstack(
        [np.eye(A.shape[0]), np.zeros((A.shape[0], Bw.shape[1] + Bd.shape[1]))]
    )
    scV = x00.T @ (sc * P) @ x00
    d00 = cp.hstack(
        [np.zeros((Bd.shape[1], A.shape[1] + Bw.shape[1])), np.eye(Bd.shape[1])]
    )
    sddTd = d00.T @ (sd * np.eye(Ded.shape[1])) @ d00
    z = cp.bmat(
        [
            [Cv, Dvw, Dvd],
            [
                np.zeros((Dvw.shape[1], Cv.shape[1])),
                np.eye(Dvw.shape[1]),
                np.zeros((Dvw.shape[1], Dvd.shape[1])),
            ],
        ]
    )
    M = cp.bmat([[Mvv, Mvw], [Mvw.T, Mww]])
    nzTMz = -z.T @ M @ z
    X1 = nVFxwd + scV + sddTd + nzTMz

    # Constrain V(x) <= c to be within the area where vk2^T vk2 <= sat_bound_region^2
    MDeltakvv2, MDeltakvw2, MDeltakww2, _, MDeltak_constraints2 = (
        controller_nonlin_constraints(eps, bound_region=sat_bound_region)
    )
    MDeltapvv2, MDeltapvw2, MDeltapww2, _, MDeltap_constraints2 = (
        plant_uncertainty_constraints(eps, bound_region=sin_bound_region)
    )
    _, _, _, _, _, _, _, _, _, Mvv2, Mvw2, Mww2, _, _, _, _, _ = closed_loop_cstor(
        MDeltapvv2, MDeltapvw2, MDeltapww2, MDeltakvv2, MDeltakvw2, MDeltakww2, "cvxpy"
    )
    nxk = k.Ak.shape[0]
    nxp = p.Ap.shape[0]
    nwp = MDeltapww2.shape[0]
    nwk = MDeltakww2.shape[0]
    ny = p.Cpy.shape[0]
    nd = Bd.shape[1]
    vk2 = (
        np.hstack(
            [
                np.zeros((k.Cku.shape[0], k.Dkvw.shape[0] - k.Cku.shape[0])),
                np.eye(k.Cku.shape[0]),
            ]
        )
        @ np.hstack([k.Ckv, k.Dkvw, k.Dkvy])
        @ np.bmat(
            [
                [np.zeros((nxk, nxp)), np.eye(nxk), np.zeros((nxk, nwp + nwk + nd))],
                [np.zeros((nwk, nxp + nxk + nwp)), np.eye(nwk), np.zeros((nwk, nd))],
                [p.Cpy, np.zeros((ny, nxk)), p.Dpyw, np.zeros((ny, nwk)), p.Dpyd],
            ]
        )
    )
    nvk2Tvk2 = -vk2.T @ vk2
    ssV = x00.T @ (s_sat * P) @ x00
    M2 = cp.bmat([[Mvv2, Mvw2], [Mvw2.T, Mww2]])
    nzTMz = -z.T @ M2 @ z
    X3 = nvk2Tvk2 + ssV + nzTMz

    e1 = np.zeros((P.shape[0], 1))
    e1[0] = 1.0
    constraints = (
        [
            X1 >> 0,
            s_sin * P - e1 @ e1.T >> 0,
            X3 >> 0,
            (1 - sc) * c - sd * dbar * dbar >= 0,
            sin_bound_region * sin_bound_region - s_sin * c >= 0,
            sat_bound_region * sat_bound_region - s_sat * c >= 0,
        ]
        + MDeltap_constraints
        + MDeltak_constraints
        + MDeltap_constraints2
        + MDeltak_constraints2
    )

    problem = cp.Problem(cp.Maximize((1 - sc) * c - sd * dbar * dbar), constraints)
    try:
        problem.solve(solver=cp.MOSEK)
    except Exception as e:
        print(f"infeasible: {e}")
        return False

    if problem.status in [
        cp.OPTIMAL,
        cp.UNBOUNDED,
        cp.OPTIMAL_INACCURATE,
        cp.UNBOUNDED_INACCURATE,
    ]:
        sc = sc.value
        sd = sd.value
        s_sin = s_sin.value
        s_sat = s_sat.value
        if not quiet:
            print(f"sc: {sc}, sd: {sd}")
            print(f"(1-sc)*c - sd * dbar^2 = {(1.0 - sc) * c - sd * dbar**2}")
            # print(
            # f"sat_bound_region * sat_bound_region - s_sat * c = {sat_bound_region * sat_bound_region - s_sat * c}"
            # )
        return True
        # return True
    else:
        if not quiet:
            print(f"Failed with status: {problem.status}")
        return False


def Pinv11(P):
    # Return the top left element of P^-1
    e1 = np.zeros((P.shape[0], 1))
    e1[0] = 1.0
    y = np.linalg.solve(P, e1)
    return y[0, 0]


def rfi_find_max_c_fixed_x1_dt(
    P,
    ubar,
    sat_bound_region,
    sin_bound_region,
    dbar,
    controller_nonlin_constraints,
    plant_uncertainty_constraints,
    closed_loop_cstor,
    convergence_threshold=1e-3,
    c_high=1000.0,
    **kwargs,
):
    # Bisect on c
    c_low = 0.0

    while c_high - c_low > convergence_threshold:
        c = (c_low + c_high) / 2
        success = rfi_fixed_x1_c_dt(
            P,
            ubar,
            sat_bound_region,
            sin_bound_region,
            c,
            dbar,
            controller_nonlin_constraints,
            plant_uncertainty_constraints,
            closed_loop_cstor,
            **kwargs,
        )
        if success:
            c_low = c
        else:
            c_high = c
    return c_low


def rfi_find_max_c_dt(
    P,
    ubar,
    sat_bound_regions,
    sin_bound_regions,
    dbar,
    controller_nonlin_constraints,
    plant_uncertainty_constraints,
    closed_loop_cstor,
    n_jobs=-1,
    **kwargs,
):
    def _call(sat_bound_region, sin_bound_region, sat_i, sin_j):
        c = rfi_find_max_c_fixed_x1_dt(
            P,
            ubar,
            sat_bound_region,
            sin_bound_region,
            dbar,
            controller_nonlin_constraints,
            plant_uncertainty_constraints,
            closed_loop_cstor,
            **kwargs,
        )
        return (sat_i, sin_j, c)

    inputs = [
        (sat_bound_regions[i], sin_bound_regions[j], i, j)
        for i in range(len(sat_bound_regions))
        for j in range(len(sin_bound_regions))
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_call)(s, t, i, j) for (s, t, i, j) in inputs
    )

    cs = np.zeros((len(sat_bound_regions), len(sin_bound_regions)))
    for i, j, c in results:
        cs[i, j] = c
    matprint(cs)
    maxi_flat_idx = np.argmax(cs)
    (maxi_sat_i, maxi_sin_j) = np.unravel_index(maxi_flat_idx, cs.shape)
    return (
        sat_bound_regions[maxi_sat_i],
        sin_bound_regions[maxi_sin_j],
        cs[maxi_sat_i, maxi_sin_j],
    )


def diss_fixed_x1_c_dt(
    P,
    supply_rate_type: Literal["l2_gain", "stability"],
    ubar,
    sat_bound_region,
    sin_bound_region,
    c,
    dbar,
    controller_nonlin_constraints,
    plant_uncertainty_constraints,
    closed_loop_cstor,
    eps=1e-3,
    l2_gain=None,
    quiet=False,
):
    MDeltakvv, MDeltakvw, MDeltakww, _, MDeltak_constraints = (
        controller_nonlin_constraints(eps, bound_region=sat_bound_region)
    )
    MDeltapvv, MDeltapvw, MDeltapww, _, MDeltap_constraints = (
        plant_uncertainty_constraints(eps, bound_region=sin_bound_region)
    )
    A, Bw, Bd, Cv, Dvw, Dvd, Ce, Dew, Ded, Mvv, Mvw, Mww, _, _, _, p, k = (
        closed_loop_cstor(
            MDeltapvv, MDeltapvw, MDeltapww, MDeltakvv, MDeltakvw, MDeltakww, "cvxpy"
        )
    )

    sh = cp.Variable(nonneg=True)
    s_sat = cp.Variable(nonneg=True)
    s_ss = cp.Variable(nonneg=True)

    # Core dissipativity condition: -V(x') + V + supply_rate - z^T M z >= 0
    ABwBd = cp.hstack([A, Bw, Bd])
    nVFxwd = -ABwBd.T @ P @ ABwBd
    x00 = cp.hstack(
        [np.eye(A.shape[0]), np.zeros((A.shape[0], Bw.shape[1] + Bd.shape[1]))]
    )
    V = x00.T @ P @ x00
    z = cp.bmat(
        [
            [Cv, Dvw, Dvd],
            [
                np.zeros((Dvw.shape[1], Cv.shape[1])),
                np.eye(Dvw.shape[1]),
                np.zeros((Dvw.shape[1], Dvd.shape[1])),
            ],
        ]
    )
    M = cp.bmat([[Mvv, Mvw], [Mvw.T, Mww]])
    nzTMz = -z.T @ M @ z
    nx = A.shape[0]
    nw = Bw.shape[1]
    nd = Bd.shape[1]
    CeDewDed = cp.hstack([Ce, Dew, Ded])
    match supply_rate_type:
        case "l2_gain":
            assert l2_gain is not None
            supply_rate = (
                cp.bmat(
                    [
                        [np.zeros((nx + nw, nx + nw + nd))],
                        [np.zeros((nd, nx + nw)), np.eye(nd)],
                    ]
                )
                - (1 / l2_gain**2) * CeDewDed.T @ CeDewDed
            )
        case "stability":
            supply_rate = 0
        case _:
            assert_never(supply_rate_type)
    X1 = nVFxwd + V + s_ss * supply_rate + nzTMz

    # Constrain V(x) <= c to be within the area where vk2^T vk2 <= sat_bound_region^2
    MDeltakvv2, MDeltakvw2, MDeltakww2, _, MDeltak_constraints2 = (
        controller_nonlin_constraints(eps, bound_region=sat_bound_region)
    )
    MDeltapvv2, MDeltapvw2, MDeltapww2, _, MDeltap_constraints2 = (
        plant_uncertainty_constraints(eps, bound_region=sin_bound_region)
    )
    _, _, _, _, _, _, _, _, _, Mvv2, Mvw2, Mww2, _, _, _, _, _ = closed_loop_cstor(
        MDeltapvv2, MDeltapvw2, MDeltapww2, MDeltakvv2, MDeltakvw2, MDeltakww2, "cvxpy"
    )
    nxk = k.Ak.shape[0]
    nxp = p.Ap.shape[0]
    nwp = MDeltapww2.shape[0]
    nwk = MDeltakww2.shape[0]
    ny = p.Cpy.shape[0]
    vk2 = (
        np.hstack(
            [
                np.zeros((k.Cku.shape[0], k.Dkvw.shape[0] - k.Cku.shape[0])),
                np.eye(k.Cku.shape[0]),
            ]
        )
        @ np.hstack([k.Ckv, k.Dkvw, k.Dkvy])
        @ np.bmat(
            [
                [np.zeros((nxk, nxp)), np.eye(nxk), np.zeros((nxk, nwp + nwk + nd))],
                [np.zeros((nwk, nxp + nxk + nwp)), np.eye(nwk), np.zeros((nwk, nd))],
                [p.Cpy, np.zeros((ny, nxk)), p.Dpyw, np.zeros((ny, nwk)), p.Dpyd],
            ]
        )
    )
    nvk2Tvk2 = -vk2.T @ vk2
    ssV = x00.T @ (s_sat * P) @ x00
    M2 = cp.bmat([[Mvv2, Mvw2], [Mvw2.T, Mww2]])
    nzTMz = -z.T @ M2 @ z
    X3 = nvk2Tvk2 + ssV + nzTMz

    e1 = np.zeros((P.shape[0], 1))
    e1[0] = 1.0
    constraints = (
        [
            X1 >> 0,
            sh * P - e1 @ e1.T >> 0,
            X3 >> 0,
            sin_bound_region * sin_bound_region - sh * c >= 0,
            sat_bound_region * sat_bound_region - s_sat * c >= 0,
            sh >= 0,
            s_sat >= 0,
            s_ss >= 0,
        ]
        + MDeltap_constraints
        + MDeltak_constraints
        + MDeltap_constraints2
        + MDeltak_constraints2
    )

    problem = cp.Problem(cp.Maximize(0), constraints)
    try:
        problem.solve(solver=cp.MOSEK)
    except Exception as e:
        print(f"infeasible: {e}")
        return False

    if problem.status in [
        cp.OPTIMAL,
        cp.UNBOUNDED,
        cp.OPTIMAL_INACCURATE,
        cp.UNBOUNDED_INACCURATE,
    ]:
        sh = sh.value
        s_sat = s_sat.value
        s_ss = s_ss.value
        if not quiet:
            print(f"sh: {sh}, s_sat: {s_sat}, s_ss: {s_ss}")
        return True
    else:
        if not quiet:
            print(f"Failed with status: {problem.status}")
        return False


def diss_find_max_c_fixed_x1_dt(
    P,
    supply_rate_type,
    ubar,
    sat_bound_region,
    sin_bound_region,
    dbar,
    controller_nonlin_constraints,
    plant_uncertainty_constraints,
    closed_loop_cstor,
    convergence_threshold=1e-3,
    c_high=1000.0,
    **kwargs,
):
    # Bisect on c
    c_low = 0.0

    while c_high - c_low > convergence_threshold:
        c = (c_low + c_high) / 2
        success = diss_fixed_x1_c_dt(
            P,
            supply_rate_type,
            ubar,
            sat_bound_region,
            sin_bound_region,
            c,
            dbar,
            controller_nonlin_constraints,
            plant_uncertainty_constraints,
            closed_loop_cstor,
            **kwargs,
        )
        if success:
            c_low = c
        else:
            c_high = c
    return c_low


def diss_find_max_c_dt(
    P,
    supply_rate_type,
    ubar,
    sat_bound_regions,
    sin_bound_regions,
    dbar,
    controller_nonlin_constraints,
    plant_uncertainty_constraints,
    closed_loop_cstor,
    n_jobs=-1,
    **kwargs,
):
    def _call(sat_bound_region, sin_bound_region, sat_i, sin_j):
        c = diss_find_max_c_fixed_x1_dt(
            P,
            supply_rate_type,
            ubar,
            sat_bound_region,
            sin_bound_region,
            dbar,
            controller_nonlin_constraints,
            plant_uncertainty_constraints,
            closed_loop_cstor,
            **kwargs,
        )
        return (sat_i, sin_j, c)

    inputs = [
        (sat_bound_regions[i], sin_bound_regions[j], i, j)
        for i in range(len(sat_bound_regions))
        for j in range(len(sin_bound_regions))
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_call)(s, t, i, j) for (s, t, i, j) in inputs
    )

    cs = np.zeros((len(sat_bound_regions), len(sin_bound_regions)))
    for i, j, c in results:
        cs[i, j] = c

    # matprint(cs)
    maxi_flat_idx = np.argmax(cs)
    (maxi_sat_i, maxi_sin_j) = np.unravel_index(maxi_flat_idx, cs.shape)
    return (
        sat_bound_regions[maxi_sat_i],
        sin_bound_regions[maxi_sin_j],
        cs[maxi_sat_i, maxi_sin_j],
    )


def level_set_contained(
    P, c, ubar, sat_bound_region, sin_bound_region, dbar, eps=1e-6, quiet=True
):
    s_sat = cp.Variable(nonneg=True)
    s_sin = cp.Variable(nonneg=True)

    # Constrain V(x) <= c to be within the area where vk2^T vk2 <= sat_bound_region^2
    MDeltakvv2, MDeltakvw2, MDeltakww2, _, MDeltak_constraints2 = (
        controller_nonlin_constraints(eps, bound_region=sat_bound_region)
    )
    MDeltapvv2, MDeltapvw2, MDeltapww2, _, MDeltap_constraints2 = (
        plant_uncertainty_constraints(eps, bound_region=sin_bound_region)
    )
    A, Bw, Bd, Cv, Dvw, Dvd, Ce, Dew, Ded, Mvv2, Mvw2, Mww2, _, _, _, p, k = (
        closed_loop_cstor(
            MDeltapvv2,
            MDeltapvw2,
            MDeltapww2,
            MDeltakvv2,
            MDeltakvw2,
            MDeltakww2,
            "cvxpy",
        )
    )

    nxk = k.Ak.shape[0]
    nxp = p.Ap.shape[0]
    nwp = MDeltapww2.shape[0]
    nwk = MDeltakww2.shape[0]
    ny = p.Cpy.shape[0]
    nd = Bd.shape[1]
    nw = Dvw.shape[1]
    nx = A.shape[0]
    vk21 = np.hstack(
        [
            np.zeros((k.Cku.shape[0], k.Dkvw.shape[0] - k.Cku.shape[0])),
            np.eye(k.Cku.shape[0]),
        ]
    )
    vk22 = np.hstack([k.Ckv, k.Dkvw, k.Dkvy])
    vk23 = np.bmat(
        [
            [np.zeros((nxk, nxp)), np.eye(nxk), np.zeros((nxk, nwp + nwk + nd))],
            [np.zeros((nwk, nxp + nxk + nwp)), np.eye(nwk), np.zeros((nwk, nd))],
            [p.Cpy, np.zeros((ny, nxk)), p.Dpyw, np.zeros((ny, nwk)), p.Dpyd],
        ]
    )
    vk2 = vk21 @ vk22 @ vk23
    nvk2Tvk2 = -vk2.T @ vk2
    x00 = cp.hstack(
        [np.eye(A.shape[0]), np.zeros((A.shape[0], Bw.shape[1] + Bd.shape[1]))]
    )
    ssV = x00.T @ (s_sat * P) @ x00
    M2 = cp.bmat([[Mvv2, Mvw2], [Mvw2.T, Mww2]])
    z = cp.bmat(
        [
            [Cv, Dvw, Dvd],
            [
                np.zeros((nw, nx)),
                np.eye(nw),
                np.zeros((nw, nd)),
            ],
        ]
    )
    nzTMz = -z.T @ M2 @ z
    X3 = nvk2Tvk2 + ssV + nzTMz

    e1 = np.zeros((P.shape[0], 1))
    e1[0] = 1.0

    constraints = (
        [
            s_sin * P - e1 @ e1.T >> 0,
            X3 >> 0,
            sin_bound_region * sin_bound_region - s_sin * c >= 0,
            sat_bound_region * sat_bound_region - s_sat * c >= 0,
        ]
        + MDeltap_constraints2
        + MDeltak_constraints2
    )

    problem = cp.Problem(cp.Maximize(sat_bound_region**2 - s_sat * c), constraints)
    try:
        problem.solve(solver=cp.MOSEK)
    except Exception as e:
        print(f"infeasible: {e}")
        return False

    if problem.status in [
        cp.OPTIMAL,
        cp.UNBOUNDED,
        cp.OPTIMAL_INACCURATE,
        cp.UNBOUNDED_INACCURATE,
    ]:
        if not quiet:
            s_sin = s_sin.value
            s_sat = s_sat.value
            print(f"s_sin: {s_sin}, s_sat: {s_sat}")
            print(
                f"sat_bound_region**2 - s_sat * c = {sat_bound_region**2 - s_sat * c}"
            )
        return True
    else:
        if not quiet:
            print(f"Failed with status: {problem.status}")
        return False


t0 = time.perf_counter()
# ubar = 20.0
# dbar = 0.0
# sat_bound_regions = np.linspace(1.0, 5.0, num=50)
# sin_bound_regions = np.linspace(1e-3, np.pi, num=31)

# P, controller_nonlin_constraints, plant_uncertainty_constraints, closed_loop_cstor = (
#     flexible_rod_setup(saturation_absolute=ubar)
# )

# maximizing_sat_bound_region, maximizing_sin_bound_region, max_c = rfi_find_max_c_dt(
#     P,
#     ubar,
#     sat_bound_regions,
#     sin_bound_regions,
#     dbar,
#     controller_nonlin_constraints,
#     plant_uncertainty_constraints,
#     closed_loop_cstor,
#     quiet=True,
#     convergence_threshold=1e-5,
# )
# print(
#     f"RFI: Maximum c found: {max_c} at sat bound region {maximizing_sat_bound_region} and sin bound region {maximizing_sin_bound_region}"
# )

# result = level_set_contained(P, 139, ubar, 2.3, 0.0, dbar, quiet=False)
# print(f"Level set contained? {result}")

# maximizing_sat_bound_region, maximizing_sin_bound_region, max_c = diss_find_max_c_dt(
#     P,
#     None,
#     "stability",
#     ubar,
#     sat_bound_regions,
#     sin_bound_regions,
#     dbar,
#     controller_nonlin_constraints,
#     plant_uncertainty_constraints,
#     closed_loop_cstor,
#     quiet=True,
#     convergence_threshold=1e-5,
# )
# print(
#     f"Diss: Maximum c found: {max_c} at sat bound region {maximizing_sat_bound_region} and sin bound region {maximizing_sin_bound_region}"
# )


saturation_rel_mgl = 1.02
ubar = saturation_rel_mgl * (0.15 * 9.81 * 0.5)
dbar = 0.1 * ubar
sat_bound_regions = np.linspace(1.0, 5.0, num=50)
sin_bound_regions = np.linspace(1e-3, np.pi, num=31)

P, controller_nonlin_constraints, plant_uncertainty_constraints, closed_loop_cstor = (
    pendulum_setup(
        "model_uncertainty",
        saturation_rel_mgl=saturation_rel_mgl,
        uncertainty_bound=0.25,
    )
)

# result = level_set_contained(P, 0.0205, ubar, 1.816, 2.73, dbar, quiet=False)
# print(f"Level set contained? {result}")

# maximizing_sat_bound_region, maximizing_sin_bound_region, max_c = rfi_find_max_c_dt(
#     P,
#     ubar,
#     sat_bound_regions,
#     sin_bound_regions,
#     dbar,
#     controller_nonlin_constraints,
#     plant_uncertainty_constraints,
#     closed_loop_cstor,
#     quiet=False,
#     convergence_threshold=1e-4,
# )
# print(
#     f"RFI: Maximum c found: {max_c} at sat bound region {maximizing_sat_bound_region} and sin bound region {maximizing_sin_bound_region}"
# )

# l2_gain = 10.0
# l2_gain = 1000.0
maximizing_sat_bound_region, maximizing_sin_bound_region, max_c = diss_find_max_c_dt(
    P,
    "stability",
    ubar,
    sat_bound_regions,
    sin_bound_regions,
    dbar,
    controller_nonlin_constraints,
    plant_uncertainty_constraints,
    closed_loop_cstor,
    quiet=True,
    convergence_threshold=1e-4,
    # l2_gain=l2_gain,
)
print(
    f"Diss: Maximum c found: {max_c} at sat bound region {maximizing_sat_bound_region} and sin bound region {maximizing_sin_bound_region}"
)

tf = time.perf_counter()
print(f"Total time: {tf - t0:.4f} seconds")
