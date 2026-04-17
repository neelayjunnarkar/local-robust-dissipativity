from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import numpy.typing as npt

# import torch

NDArrayF32 = npt.NDArray[np.float32]


def is_positive_semidefinite(X):
    if not np.allclose(X, X.T):
        return False
    eigvals, _eigvecs = np.linalg.eigh(X)
    if np.min(eigvals) < 0:
        return False, f"Minimum eigenvalue {np.min(eigvals)} < 0"
    return True


def is_positive_definite(X):
    # Check symmetric.
    if not np.allclose(X, X.T):
        return False
    # Check PD (np.linalg.cholesky does not check for symmetry)
    try:
        np.linalg.cholesky(X)
    except Exception as _e:
        return False
    return True


@dataclass
class PlantParameters:
    Ap: Any  # NDArrayF32
    Bpw: Any  # NDArrayF32
    Bpd: Any  # NDArrayF32
    Bpu: Any  # NDArrayF32
    Cpv: Any  # NDArrayF32
    Dpvw: Any  # NDArrayF32
    Dpvd: Any  # NDArrayF32
    Dpvu: Any  # NDArrayF32
    Cpe: Any  # NDArrayF32
    Dpew: Any  # NDArrayF32
    Dped: Any  # NDArrayF32
    Dpeu: Any  # NDArrayF32
    Cpy: Any  # NDArrayF32
    Dpyw: Any  # NDArrayF32
    Dpyd: Any  # NDArrayF32
    MDeltapvv: Any = None  # NDArrayF32
    MDeltapvw: Any = None  # NDArrayF32
    MDeltapww: Any = None  # NDArrayF32
    Xdd: Any = None
    Xde: Any = None
    Xee: Any = None

    def np_to_torch(self, device):
        Ap = from_numpy(self.Ap, device=device)
        Bpw = from_numpy(self.Bpw, device=device)
        Bpd = from_numpy(self.Bpd, device=device)
        Bpu = from_numpy(self.Bpu, device=device)
        Cpv = from_numpy(self.Cpv, device=device)
        Dpvw = from_numpy(self.Dpvw, device=device)
        Dpvd = from_numpy(self.Dpvd, device=device)
        Dpvu = from_numpy(self.Dpvu, device=device)
        Cpe = from_numpy(self.Cpe, device=device)
        Dpew = from_numpy(self.Dpew, device=device)
        Dped = from_numpy(self.Dped, device=device)
        Dpeu = from_numpy(self.Dpeu, device=device)
        Cpy = from_numpy(self.Cpy, device=device)
        Dpyw = from_numpy(self.Dpyw, device=device)
        Dpyd = from_numpy(self.Dpyd, device=device)
        # fmt: off
        MDeltapvv = from_numpy(self.MDeltapvv, device=device) if self.MDeltapvv is not None else None
        MDeltapvw = from_numpy(self.MDeltapvw, device=device) if self.MDeltapvw is not None else None
        MDeltapww = from_numpy(self.MDeltapww, device=device) if self.MDeltapww is not None else None
        Xdd = from_numpy(self.Xdd, device=device) if self.Xdd is not None else None
        Xde = from_numpy(self.Xde, device=device) if self.Xde is not None else None
        Xee = from_numpy(self.Xee, device=device) if self.Xee is not None else None
        return PlantParameters(
            Ap, Bpw, Bpd, Bpu, Cpv, Dpvw, Dpvd, Dpvu,
            Cpe, Dpew, Dped, Dpeu, Cpy, Dpyw, Dpyd,
            MDeltapvv, MDeltapvw, MDeltapww,
            Xdd, Xde, Xee
        )
        # fmt: on


@dataclass
class ControllerThetaParameters:
    Ak: Any  # NDArrayF32
    Bkw: Any  # NDArrayF32
    Bky: Any  # NDArrayF32
    Ckv: Any  # NDArrayF32
    Dkvw: Any  # NDArrayF32
    Dkvy: Any  # NDArrayF32
    Cku: Any  # NDArrayF32
    Dkuw: Any  # NDArrayF32
    Dkuy: Any  # NDArrayF32
    # Lambda: Any = None  # NDArrayF32
    MDeltakvv: Any = None
    MDeltakvw: Any = None
    MDeltakww: Any = None

    def torch_to_np(self):
        Ak = to_numpy(self.Ak)
        Bkw = to_numpy(self.Bkw)
        Bky = to_numpy(self.Bky)
        Ckv = to_numpy(self.Ckv)
        Dkvw = to_numpy(self.Dkvw)
        Dkvy = to_numpy(self.Dkvy)
        Cku = to_numpy(self.Cku)
        Dkuw = to_numpy(self.Dkuw)
        Dkuy = to_numpy(self.Dkuy)
        # Lambda = to_numpy(self.Lambda) if self.Lambda is not None else None
        return ControllerThetaParameters(
            Ak,
            Bkw,
            Bky,
            Ckv,
            Dkvw,
            Dkvy,
            Cku,
            Dkuw,
            Dkuy,  # , Lambda
        )

    def np_to_torch(self, device):
        Ak = from_numpy(self.Ak, device=device)
        Bkw = from_numpy(self.Bkw, device=device)
        Bky = from_numpy(self.Bky, device=device)
        Ckv = from_numpy(self.Ckv, device=device)
        Dkvw = from_numpy(self.Dkvw, device=device)
        Dkvy = from_numpy(self.Dkvy, device=device)
        Cku = from_numpy(self.Cku, device=device)
        Dkuw = from_numpy(self.Dkuw, device=device)
        Dkuy = from_numpy(self.Dkuy, device=device)
        Lambda = (
            from_numpy(self.Lambda, device=device) if self.Lambda is not None else None
        )
        return ControllerThetaParameters(
            Ak, Bkw, Bky, Ckv, Dkvw, Dkvy, Cku, Dkuw, Dkuy, Lambda
        )

    def matrix(self, type="np"):
        if type == "np":
            stacker = np
        elif type == "torch":
            stacker = torch
        else:
            raise ValueError(f"Unexpected type: {type}.")
        # fmt: off
        theta = stacker.vstack((
            stacker.hstack((self.Ak,  self.Bkw, self.Bky)),
            stacker.hstack((self.Ckv, self.Dkvw, self.Dkvy)),
            stacker.hstack((self.Cku, self.Dkuw, self.Dkuy))
        ))
        # fmt: on
        return theta

    def torch_construct_thetahat(self, P, plant_params: PlantParameters, eps=1e-3):
        """Construct thetahat parameters from self and P"""
        state_size = self.Ak.shape[0]
        assert state_size == plant_params.Ap.shape[0]

        S = P[:state_size, :state_size]
        S = 0.5 * (S + S.t())
        U = P[:state_size, state_size:]
        assert torch.allclose(S, S.t()), "S is not symmetric"
        assert torch.linalg.eigvalsh(S).min() > 0, (
            f"S min eigval is {torch.linalg.eigvalsh(S).min()}"
        )

        # Pinv = self.P.inverse()
        # R = Pinv[:state_size, :state_size]
        # V = Pinv[:state_size, state_size:]
        # if (not torch.allclose(R, R.t())) and  (R - R.t()).abs().max() < eps:
        #     print(f"R: {torch.allclose(R, R.t())}, {(R - R.t()).abs().max()}, {torch.linalg.eigvalsh(R).min()}")
        #     R = (R + R.t())/2.0
        #     print(f"R: {torch.allclose(R, R.t())}, {(R - R.t()).abs().max()}, {torch.linalg.eigvalsh(R).min()}")
        # assert torch.allclose(R, R.t()), "R is not symmetric"
        # assert torch.linalg.eigvalsh(R).min() > eps, f"P min eigval is {torch.linalg.eigvalsh(R).min()}"
        # Solve P Y = Y2 for Y
        # fmt: off
        Y2 = torch.vstack((
            torch.hstack((torch.eye(S.shape[0], device=S.device), S)),
            torch.hstack((S.new_zeros((U.t().shape[0], S.shape[0])), U.t()))
        ))
        # fmt: on
        Y = torch.linalg.solve(P, Y2)
        R = Y[:state_size, :state_size]
        R = 0.5 * (R + R.t())
        V = Y[state_size:, :state_size].t()
        # fmt: off
        # if (not torch.allclose(R, R.t())) and  (R - R.t()).abs().max() < eps:
        #     print(f"R: {torch.allclose(R, R.t())}, {(R - R.t()).abs().max()}, {torch.linalg.eigvalsh(R).min()}")
        #     R = (R + R.t())/2.0
        #     print(f"R: {torch.allclose(R, R.t())}, {(R - R.t()).abs().max()}, {torch.linalg.eigvalsh(R).min()}")
        # fmt: on
        assert torch.allclose(R, R.t()), "R is not symmetric"
        assert torch.linalg.eigvalsh(R).min() > 0, (
            f"P min eigval is {torch.linalg.eigvalsh(R).min()}"
        )

        Lambda = self.Lambda
        Lambda = 0.5 * (Lambda + Lambda.t())

        # NA = NA1 + NA2 NA3 NA4

        Ap = plant_params.Ap
        Bpu = plant_params.Bpu
        Cpy = plant_params.Cpy

        NA111 = torch.mm(S, torch.mm(Ap, R))
        NA112 = V.new_zeros((state_size, Cpy.shape[0]))
        NA121 = V.new_zeros((Bpu.shape[1], state_size))
        NA122 = V.new_zeros((Bpu.shape[1], Cpy.shape[0]))
        NA1 = torch.vstack((torch.hstack((NA111, NA112)), torch.hstack((NA121, NA122))))

        NA211 = U
        NA212 = torch.mm(S, Bpu)
        NA221 = V.new_zeros((Bpu.shape[1], U.shape[1]))
        NA222 = torch.eye(Bpu.shape[1], device=V.device)
        NA2 = torch.vstack((torch.hstack((NA211, NA212)), torch.hstack((NA221, NA222))))

        # fmt: off
        NA3 = torch.vstack((
            torch.hstack((self.Ak, self.Bky)),
            torch.hstack((self.Cku, self.Dkuy))
        ))
        # fmt: on

        NA411 = V.t()
        NA412 = V.new_zeros(V.shape[1], Cpy.shape[0])
        NA421 = torch.mm(Cpy, R)
        NA422 = torch.eye(Cpy.shape[0], device=V.device)
        NA4 = torch.vstack((torch.hstack((NA411, NA412)), torch.hstack((NA421, NA422))))

        NA = NA1 + torch.mm(NA2, torch.mm(NA3, NA4))
        NA11 = NA[:state_size, :state_size]
        NA12 = NA[:state_size, state_size:]
        NA21 = NA[state_size:, :state_size]
        NA22 = NA[state_size:, state_size:]

        NB = torch.mm(S, torch.mm(Bpu, self.Dkuw)) + torch.mm(U, self.Bkw)
        # fmt: off
        NC = torch.mm(Lambda, torch.mm(self.Dkvy, torch.mm(Cpy, R))) \
            + torch.mm(Lambda, torch.mm(self.Ckv, V.t()))
        # fmt: on

        Dvyhat = torch.mm(Lambda, self.Dkvy)
        Dvwhat = torch.mm(Lambda, self.Dkvw)

        return ControllerThetahatParameters(
            S, R, NA11, NA12, NA21, NA22, NB, NC, self.Dkuw, Dvyhat, Dvwhat, Lambda
        )

    def wrap_in_saturation(self, saturation):
        # Return a new model that applies a saturation to the control input
        # and represents this by adding (vk2 = Delta(wk2)), where this
        # Delta applies saturation in [-1, 1], and appropriate gains are placed so
        # u -> utilde is utilde = saturation * sat(u / saturation)
        nx = self.Ak.shape[0]
        nw1 = self.Bkw.shape[1]
        ny = self.Bky.shape[1]
        nv1 = self.Ckv.shape[0]
        nu = self.Cku.shape[0]
        assert nu == 1, "Only supports nu = 1"
        return ControllerThetaParameters(
            Ak=self.Ak,
            Bkw=np.hstack([self.Bkw, np.zeros((nx, 1))]),
            Bky=self.Bky,
            Ckv=np.vstack([self.Ckv, self.Cku / saturation]),
            Dkvw=np.hstack(
                [
                    np.vstack([self.Dkvw, self.Dkuw / saturation]),
                    np.zeros((nv1 + nu, 1)),
                ]
            ),
            Dkvy=np.vstack([self.Dkvy, self.Dkuy / saturation]),
            Cku=np.zeros((1, nx)),
            Dkuw=np.hstack([np.zeros((1, nw1)), np.array([[saturation]])]),
            Dkuy=np.zeros((1, ny)),
        )


@dataclass
class ControllerThetahatParameters:
    S: Any
    R: Any
    NA11: Any
    NA12: Any
    NA21: Any
    NA22: Any
    NB: Any
    NC: Any
    Dkuw: Any
    Dkvyhat: Any
    Dkvwhat: Any
    Lambda: Any

    def torch_to_np(self):
        S = to_numpy(self.S)
        R = to_numpy(self.R)
        NA11 = to_numpy(self.NA11)
        NA12 = to_numpy(self.NA12)
        NA21 = to_numpy(self.NA21)
        NA22 = to_numpy(self.NA22)
        NB = to_numpy(self.NB)
        NC = to_numpy(self.NC)
        Dkuw = to_numpy(self.Dkuw)
        Dkvyhat = to_numpy(self.Dkvyhat)
        Dkvwhat = to_numpy(self.Dkvwhat)
        Lambda = to_numpy(self.Lambda)
        return ControllerThetahatParameters(
            S, R, NA11, NA12, NA21, NA22, NB, NC, Dkuw, Dkvyhat, Dkvwhat, Lambda
        )

    def np_to_torch(self, device):
        S = from_numpy(self.S, device=device)
        R = from_numpy(self.R, device=device)
        NA11 = from_numpy(self.NA11, device=device)
        NA12 = from_numpy(self.NA12, device=device)
        NA21 = from_numpy(self.NA21, device=device)
        NA22 = from_numpy(self.NA22, device=device)
        NB = from_numpy(self.NB, device=device)
        NC = from_numpy(self.NC, device=device)
        Dkuw = from_numpy(self.Dkuw, device=device)
        Dkvyhat = from_numpy(self.Dkvyhat, device=device)
        Dkvwhat = from_numpy(self.Dkvwhat, device=device)
        Lambda = from_numpy(self.Lambda, device=device)
        return ControllerThetahatParameters(
            S, R, NA11, NA12, NA21, NA22, NB, NC, Dkuw, Dkvyhat, Dkvwhat, Lambda
        )

    def torch_construct_theta(self, plant_params: PlantParameters, eps=1e-3):
        """Construct theta, the parameters of the controller, from thetahat,
        the decision variables for the dissipativity condition."""

        state_size = self.S.shape[0]
        assert state_size == plant_params.Ap.shape[0]

        svdU, svdSigma, svdV_T = torch.linalg.svd(
            torch.eye(state_size) - self.R @ self.S
        )
        sqrt_svdSigma = torch.diag(torch.sqrt(svdSigma))
        V = svdU @ sqrt_svdSigma
        U = svdV_T.t() @ sqrt_svdSigma

        # V = self.R
        # U = torch.linalg.solve(
        #     V, torch.eye(self.R.shape[0], device=V.device) - torch.mm(self.R, self.S)
        # ).t()

        # Construct P via P = [I, S; 0, U^T] Y^-1
        # fmt: off
        Y = torch.vstack((
            torch.hstack((self.R, torch.eye(self.R.shape[0], device=self.R.device))),
            torch.hstack((V.t(), V.new_zeros((V.t().shape[0], self.R.shape[0]))))
        ))
        Y2 = torch.vstack((
            torch.hstack((torch.eye(self.S.shape[0], device=self.S.device), self.S)),
            torch.hstack((V.new_zeros((U.t().shape[0], self.S.shape[0])), U.t()))
        ))
        # fmt: on
        P = torch.linalg.solve(Y.t(), Y2.t())
        P = 0.5 * (P + P.t())
        assert torch.allclose(P, P.t()), "R is not symmetric"
        assert torch.linalg.eigvalsh(P).min() > 0, (
            f"P min eigval is {torch.linalg.eigvalsh(P).min()}"
        )

        # Reconstruct Dvy and Dvw from Dvyhat and Dvwhat
        Dvy = torch.linalg.solve(self.Lambda, self.Dkvyhat)
        Dvw = torch.linalg.solve(self.Lambda, self.Dkvwhat)

        Ap = plant_params.Ap
        Bpu = plant_params.Bpu
        Cpy = plant_params.Cpy

        Duy = self.NA22
        By = torch.linalg.solve(U, self.NA12 - self.S @ Bpu @ Duy)
        Cu = torch.linalg.solve(V, self.NA21.t() - self.R @ Cpy.t() @ Duy.t()).t()
        # fmt: off
        AVT = torch.linalg.solve(
            U,
            self.NA11 \
                - U @ By @ Cpy @ self.R \
                - self.S @ Bpu @ (Cu @ V.t() + Duy @ Cpy @ self.R) \
                - self.S @ Ap @ self.R
        )
        # fmt: on
        A = torch.linalg.solve(V, AVT.t()).t()

        # Solve for Bw from NB
        # NB = NB1 + U Bw
        NB1 = torch.mm(self.S, torch.mm(Bpu, self.Dkuw))
        Bw = torch.linalg.solve(U, self.NB - NB1)

        # Solve for Cv from NC
        # NC = NC1 + Lambda Cv V^T
        NC1 = torch.mm(self.Dkvyhat, torch.mm(Cpy, self.R))
        CvVT = torch.linalg.solve(self.Lambda, self.NC - NC1)
        Cv = torch.linalg.solve(V, CvVT.t()).t()

        # Bring together the theta parameters here
        theta = ControllerThetaParameters(
            A, Bw, By, Cv, Dvw, Dvy, Cu, self.Dkuw, Duy, self.Lambda
        )

        return theta, P


@dataclass
class ControllerLTIThetaParameters:
    Ak: Any  # NDArrayF32
    Bky: Any  # NDArrayF32
    Cku: Any  # NDArrayF32
    Dkuy: Any  # NDArrayF32

    def torch_to_np(self):
        Ak = to_numpy(self.Ak)
        Bky = to_numpy(self.Bky)
        Cku = to_numpy(self.Cku)
        Dkuy = to_numpy(self.Dkuy)
        return ControllerLTIThetaParameters(Ak, Bky, Cku, Dkuy)

    def np_to_torch(self, device):
        Ak = from_numpy(self.Ak, device=device)
        Bky = from_numpy(self.Bky, device=device)
        Cku = from_numpy(self.Cku, device=device)
        Dkuy = from_numpy(self.Dkuy, device=device)
        return ControllerLTIThetaParameters(Ak, Bky, Cku, Dkuy)

    def torch_construct_thetahat(self, P, plant_params: PlantParameters):
        state_size = self.Ak.shape[0]
        input_size = self.Bky.shape[1]
        output_size = self.Cku.shape[0]
        nonlin_size = 1

        nonlin_theta = ControllerThetaParameters(
            Ak=self.Ak,
            Bkw=self.Ak.new_zeros((state_size, nonlin_size)),
            Bky=self.Bky,
            Ckv=self.Ak.new_zeros((nonlin_size, state_size)),
            Dkvw=self.Ak.new_zeros((nonlin_size, nonlin_size)),
            Dkvy=self.Ak.new_zeros((nonlin_size, input_size)),
            Cku=self.Cku,
            Dkuw=self.Ak.new_zeros((output_size, nonlin_size)),
            Dkuy=self.Dkuy,
            Lambda=self.Ak.new_zeros((nonlin_size, nonlin_size)),
        )
        nonlin_thetahat = nonlin_theta.torch_construct_thetahat(P, plant_params)

        thetahat = ControllerLTIThetahatParameters(
            nonlin_thetahat.S,
            nonlin_thetahat.R,
            nonlin_thetahat.NA11,
            nonlin_thetahat.NA12,
            nonlin_thetahat.NA21,
            nonlin_thetahat.NA22,
        )
        return thetahat


@dataclass
class ControllerLTIThetahatParameters:
    S: Any
    R: Any
    NA11: Any
    NA12: Any
    NA21: Any
    NA22: Any

    def torch_to_np(self):
        S = to_numpy(self.S)
        R = to_numpy(self.R)
        NA11 = to_numpy(self.NA11)
        NA12 = to_numpy(self.NA12)
        NA21 = to_numpy(self.NA21)
        NA22 = to_numpy(self.NA22)
        return ControllerLTIThetahatParameters(S, R, NA11, NA12, NA21, NA22)

    def np_to_torch(self, device):
        S = from_numpy(self.S, device=device)
        R = from_numpy(self.R, device=device)
        NA11 = from_numpy(self.NA11, device=device)
        NA12 = from_numpy(self.NA12, device=device)
        NA21 = from_numpy(self.NA21, device=device)
        NA22 = from_numpy(self.NA22, device=device)
        return ControllerLTIThetahatParameters(S, R, NA11, NA12, NA21, NA22)

    def torch_construct_theta(self, plant_params: PlantParameters):
        """Construct theta, the parameters of the controller, from thetahat,
        the decision variables for the dissipativity condition."""
        state_size = self.S.shape[0]

        # np.linalg.solve(A, B) solves for X in AX = B, and assumes A is invertible

        # Construct V and U by solving VU^T = I - RS
        svdU, svdSigma, svdV_T = torch.linalg.svd(
            torch.eye(state_size) - self.R @ self.S
        )
        sqrt_svdSigma = torch.diag(torch.sqrt(svdSigma))
        V = svdU @ sqrt_svdSigma
        U = svdV_T.t() @ sqrt_svdSigma

        # Construct P via P = [I, S; 0, U^T] Y^-1
        # fmt: off
        Y = torch.vstack((
            torch.hstack((self.R, torch.eye(self.R.shape[0]))),
            torch.hstack((V.t(), torch.zeros((V.t().shape[0], self.R.shape[0]))))
        ))
        Y2 = torch.vstack((
            torch.hstack((torch.eye(self.S.shape[0]), self.S)),
            torch.hstack((torch.zeros((U.t().shape[0], self.S.shape[0])), U.t()))
        ))
        # fmt: on
        P = torch.linalg.solve(Y.t(), Y2.t())
        P = 0.5 * (P + P.t())

        Ap = plant_params.Ap
        Bpu = plant_params.Bpu
        Cpy = plant_params.Cpy

        Duy = self.NA22
        By = torch.linalg.solve(U, self.NA12 - self.S @ Bpu @ Duy)
        Cu = torch.linalg.solve(V, self.NA21.t() - self.R @ Cpy.t() @ Duy.t()).t()
        AVT = torch.linalg.solve(
            U,
            self.NA11
            - U @ By @ Cpy @ self.R
            - self.S @ Bpu @ (Cu @ V.t() + Duy @ Cpy @ self.R)
            - self.S @ Ap @ self.R,
        )
        A = torch.linalg.solve(V, AVT.t()).t()

        controller = ControllerLTIThetaParameters(Ak=A, Bky=By, Cku=Cu, Dkuy=Duy)
        return controller, P


def construct_closed_loop(
    plant_params: PlantParameters,
    controller_params: ControllerThetaParameters,
    stacker,
    LDeltap=None,
):
    Ap = plant_params.Ap
    Bpw = plant_params.Bpw
    Bpd = plant_params.Bpd
    Bpu = plant_params.Bpu
    Cpv = plant_params.Cpv
    Dpvw = plant_params.Dpvw
    Dpvd = plant_params.Dpvd
    Dpvu = plant_params.Dpvu
    Cpe = plant_params.Cpe
    Dpew = plant_params.Dpew
    Dped = plant_params.Dped
    Dpeu = plant_params.Dpeu
    Cpy = plant_params.Cpy
    Dpyw = plant_params.Dpyw
    Dpyd = plant_params.Dpyd
    MDeltapvv = plant_params.MDeltapvv
    MDeltapvw = plant_params.MDeltapvw
    MDeltapww = plant_params.MDeltapww
    Ak = controller_params.Ak
    Bkw = controller_params.Bkw
    Bky = controller_params.Bky
    Ckv = controller_params.Ckv
    Dkvw = controller_params.Dkvw
    Dkvy = controller_params.Dkvy
    Cku = controller_params.Cku
    Dkuw = controller_params.Dkuw
    Dkuy = controller_params.Dkuy
    MDeltakvv = controller_params.MDeltakvv
    MDeltakvw = controller_params.MDeltakvw
    MDeltakww = controller_params.MDeltakww
    # Lambda = controller_params.Lambda

    if stacker == "numpy":
        stacker = np.bmat
    elif stacker == "cvxpy":
        stacker = cp.bmat
    else:
        raise ValueError(f"Stacker {stacker} must be 'numpy' or 'cvxpy'.")

    # fmt: off
    A = stacker([
        [Ap + Bpu @ Dkuy @ Cpy, Bpu @ Cku],
        [Bky @ Cpy, Ak]
    ])
    Bw = stacker([
        [Bpw + Bpu @ Dkuy @ Dpyw, Bpu @ Dkuw],
        [Bky @ Dpyw, Bkw]
    ])
    Bd = stacker([
        [Bpd + Bpu @ Dkuy @ Dpyd],
        [Bky @ Dpyd]
    ])
    Cv = stacker([
        [Cpv + Dpvu @ Dkuy @ Cpy, Dpvu @ Cku],
        [Dkvy @ Cpy, Ckv]
    ])
    Dvw = stacker([
        [Dpvw + Dpvu @ Dkuy @ Dpyw, Dpvu @ Dkuw],
        [Dkvy @ Dpyw, Dkvw]
    ])
    Dvd = stacker([
        [Dpvd + Dpvu @ Dkuy @ Dpyd],
        [Dkvy @ Dpyd]
    ])
    Ce = stacker([
        [Cpe + Dpeu @ Dkuy @ Cpy, Dpeu @ Cku]
    ])
    Dew = stacker([
        [Dpew + Dpeu @ Dkuy @ Dpyw, Dpeu @ Dkuw]
    ])
    Ded = stacker([
        [Dped + Dpeu @ Dkuy @ Dpyd]
    ])

    # if LDeltap is not None:
    #     LDelta = stacker([
    #         [LDeltap, np.zeros((LDeltap.shape[0], Lambda.shape[1]))],
    #         [np.zeros((Lambda.shape[0], LDeltap.shape[1] + Lambda.shape[1]))]
    #     ])
    # else:
    #     LDelta = None

    Mvv = stacker([
        [MDeltapvv, np.zeros((MDeltapvv.shape[0], MDeltakvv.shape[1]))],
        [np.zeros((MDeltakvv.shape[0], MDeltapvv.shape[1])), MDeltakvv]
    ])
    Mvw = stacker([
        [MDeltapvw, np.zeros((MDeltapvv.shape[0], MDeltakvw.shape[1]))],
        [np.zeros((MDeltakvw.shape[0], MDeltapvw.shape[1])), MDeltakvw]
    ])
    Mww = stacker([
        [MDeltapww, np.zeros((MDeltapww.shape[0], MDeltakww.shape[1]))],
        [np.zeros((MDeltakww.shape[0], MDeltapww.shape[1])), MDeltakww]
    ])
    # fmt: on

    return A, Bw, Bd, Cv, Dvw, Dvd, Ce, Dew, Ded, Mvv, Mvw, Mww


def MDeltavv_to_LDelta(MDeltavv):
    assert is_positive_semidefinite(MDeltavv)
    Dm, Vm = np.linalg.eigh(MDeltavv)
    LDelta = np.diag(np.sqrt(Dm)) @ Vm.T
    return LDelta


def from_numpy(array, device=None):
    out = torch.tensor(array.astype(np.float32))
    if device is not None:
        return out.to(device)
    else:
        return out


def to_numpy(tensor):
    return tensor.to("cpu").detach().numpy()


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


def project_mat(P, n):
    # Print onto first n coordinates
    P11 = P[:n, :n]
    P12 = P[:n, n:]
    P22 = P[n:, n:]

    P22invP12T = np.linalg.solve(P22, P12.T)
    Phat = P11 - P12 @ P22invP12T

    return Phat


def plot_levelset(
    P,
    c=1.0,
    ax=None,
    npts=300,
    show_axes=True,
    ellipse_kwargs=None,
    axes_kwargs=None,
):
    """
    Plot the level set x^T P x = c.

    Args:
        P (array_like): 2x2 symmetric positive-definite matrix.
        c (float): positive scalar specifying the levelset x^T P x = c.
        ax (matplotlib.axes.Axes, optional): axes to draw on. If None, creates new figure/axes.
        npts (int): number of points used to draw the ellipse.
        show_axes (bool): whether to set equal aspect and show axis lines.
        plot_principal_axes (bool): whether to draw principal axes (eigenvectors scaled).
        ellipse_kwargs (dict): kwargs forwarded to plot() for the ellipse line.
        axes_kwargs (dict): kwargs forwarded to plot() for principal axes lines.
    Returns:
        ax: The matplotlib Axes with the plot.
    """
    import matplotlib.pyplot as plt

    P = np.asarray(P, dtype=float)
    if P.shape != (2, 2):
        raise ValueError("P must be a 2x2 matrix.")
    if c <= 0:
        raise ValueError("c must be positive.")

    # Ensure symmetry (tolerant)
    if not np.allclose(P, P.T, atol=1e-8):
        P = 0.5 * (P + P.T)

    # Eigen-decomposition (guarantees real eigenvalues for symmetric P)
    eigvals, eigvecs = np.linalg.eigh(P)
    if np.any(eigvals <= 0):
        raise ValueError("P must be positive definite (all eigenvalues > 0).")

    # radii along principal axes: sqrt(c / lambda_i)
    radii = np.sqrt(c / eigvals)  # shape (2,)

    # Parameterize the unit circle
    theta = np.linspace(0, 2 * np.pi, npts)
    unit_circle = np.vstack((np.cos(theta), np.sin(theta)))  # shape (2, npts)

    # Map unit circle to ellipse: x = Q * diag(radii) * unit_circle
    ellipse_points = eigvecs @ (np.diag(radii) @ unit_circle)  # shape (2, npts)
    xs, ys = ellipse_points[0, :], ellipse_points[1, :]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ellipse_kwargs = ellipse_kwargs or {}
    ax.plot(
        xs,
        ys,
        **{
            **{"color": "C0", "linewidth": 2, "label": f"$x^T P x = {c}$"},
            **ellipse_kwargs,
        },
    )

    if show_axes:
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.set_aspect("equal", adjustable="box")

        # set limits with a small margin
        max_radius = np.max(radii)
        # orient margin by the ellipse extents in x and y
        margin = max_radius * 1.2
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)

    ax.legend(loc="best")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Level set $x^T P x = {c}$")

    return ax
