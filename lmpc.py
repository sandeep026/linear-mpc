from typing import Any

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveInt,
    field_validator,
    model_validator,
)
from scipy import sparse


class LinearMPC(BaseModel):
    intervals: PositiveInt
    nx: PositiveInt
    nu: PositiveInt

    A: np.ndarray
    B: np.ndarray

    Q: np.ndarray
    R: np.ndarray
    Qend: np.ndarray

    x0: np.ndarray
    x_ref: np.ndarray
    u_ref: np.ndarray

    x_lb: np.ndarray
    x_ub: np.ndarray
    u_lb: np.ndarray
    u_ub: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("A", "B", "Q", "R", "Qend",
                     "x0", "x_ref", "u_ref",
                     "x_lb", "x_ub", "u_lb", "u_ub",
                     mode="before")
    @classmethod
    def to_numpy(cls, mat):
        return np.asarray(mat, dtype=np.float64)

    @field_validator("Q", "Qend")
    @classmethod
    def check_psd(cls, mat):
        if not np.allclose(mat, mat.T, atol=1e-10, rtol=0):
            raise ValueError("Q or Qend must be symmetric")

        if np.min(np.linalg.eigvalsh(mat)) < -1e-10:
            raise ValueError(
                "Q or Qend must be positive semidefinite"
            )

        return mat

    @field_validator("R")
    @classmethod
    def check_pd(cls, mat):
        if not np.allclose(mat, mat.T, atol=1e-10, rtol=0):
            raise ValueError("R must be symmetric")

        if np.min(np.linalg.eigvalsh(mat)) <= 1e-10:
            raise ValueError(
                "R must be positive definite"
            )

        return mat

    @model_validator(mode="after")
    def validate_mpc(self):
        n = self.intervals
        nx, nu = self.nx, self.nu

        shapes = {
            "A": (nx, nx),
            "B": (nx, nu),
            "Q": (nx, nx),
            "R": (nu, nu),
            "Qend": (nx, nx),
            "x_lb": (nx, 1),
            "x_ub": (nx, 1),
            "u_lb": (nu, 1),
            "u_ub": (nu, 1),
            "x0": (nx, 1),
            "x_ref": (nx, n + 1),
            "u_ref": (nu, n),
        }

        for name, shape in shapes.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(
                    f"{name}: expected {shape}, got {value.shape}"
                )

        if np.any(self.x_lb > self.x_ub):
            raise ValueError(
                "lower must be <= upper bound for x"
            )

        if np.any(self.u_lb > self.u_ub):
            raise ValueError(
                "lower must be <= upper bound for u"
            )

        return self

    def generate_QP(self):
        n = self.intervals
        nx, nu = self.nx, self.nu

        A = sparse.csc_matrix(self.A)
        B = sparse.csc_matrix(self.B)
        Q = sparse.csc_matrix(self.Q)
        R = sparse.csc_matrix(self.R)
        Qend = sparse.csc_matrix(self.Qend)

        # hessian gradient
        Hqp = sparse.block_diag(
            (
                sparse.kron(
                    sparse.eye(n, format="csc"),
                    Q,
                    format="csc",
                ),
                Qend,
                sparse.kron(
                    sparse.eye(n, format="csc"),
                    R,
                    format="csc",
                ),
            ),
            format="csc",
        )

        # c = -1 H w_ref
        cqp = -1* np.concatenate(
            (
                (self.Q @ self.x_ref[:, :n])
                .reshape(-1, order="F"),

                (self.Qend @ self.x_ref[:, n])
                .reshape(-1, order="F"),

                (self.R @ self.u_ref)
                .reshape(-1, order="F"),
            )
        )

        # equality constraints

        Cx = (
            sparse.eye(
                (n + 1) * nx,
                format="csc",
            )
            - sparse.kron(
                sparse.diags(
                    np.ones(n),
                    offsets=-1,
                    shape=(n + 1, n + 1),
                    format="csc",
                ),
                A,
                format="csc",
            )
        )

        Cu = sparse.vstack(
            (
                sparse.csc_matrix((nx, n * nu)),
                sparse.kron(
                    sparse.eye(n, format="csc"),
                    -B,
                    format="csc",
                ),
            ),
            format="csc",
        )

        Aqp = sparse.hstack(
            (Cx, Cu),
            format="csc",
        )


        bqp = np.zeros(
            (n + 1) * nx,
            dtype=np.float64,
        )
        bqp[:nx] = self.x0[:, 0]


        #simple bounds
        w_lb = np.concatenate(
            (
                np.tile(self.x_lb[:, 0], n + 1),
                np.tile(self.u_lb[:, 0], n),
            )
        )

        w_ub = np.concatenate(
            (
                np.tile(self.x_ub[:, 0], n + 1),
                np.tile(self.u_ub[:, 0], n),
            )
        )

        return {
            "H": Hqp,
            "c": cqp,
            "A": Aqp,
            "b": bqp,
            "lb": w_lb,
            "ub": w_ub,
        }
