from typing import Optional
from dataclasses import dataclass
import numpy as np
from scipy import optimize  # type: ignore
from typing import Union
from numpy import float_
from numpy.typing import NDArray

FloatArray = NDArray[float_]
Floats = Union[float, FloatArray]

@dataclass
class SVI:
    """The SVI (Stochastic Volatility Inspired) model.
    The model directly represents a volatility curve by the function
      `w(x) = a + b*(rho*(x-m) + sqrt((x-m)**2 + sigma**2))`
    where
      `x` is the log-moneyness, i.e. `x = log(s/k)`,
      `w` is the total implied variance, i.e. `w = t*(iv**2)`,
    and `a`, `b >= 0`, `-1 < rho < 1`, `m`, `sigma > 0` are model parameters.
    Expiration time is assumed to be fixed and is not explicitly specified (but
    see `to_jumpwing` and `from_jumpwing` functions).
    Attributes:
      a, b, rho, m, sigma: Model parameters.
    Methods:
      calibrate: Calibrates parameters of the model.
    """
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    @staticmethod
    def _calibrate_adc(x: FloatArray, w: FloatArray, m: float,
                       sigma: float) -> float:
        """Calibrates the raw parameters `a, d, c` given `m, sigma`.
        This is an auxiliary function used in the two-step calibration
        procedure. It finds `a, d, c` which minimize the sum of squares of the
        differences of the given total implied variances and the ones produced
        by the model, assuming that `m, sigma` are given and fixed.
        Args:
          x: Array of log-moneynesses
          w: Array of total implied variances.
          m: Parameter `m` of the model.
          sigma: Parameter `sigma` of the model.
        Returns:
          Tuple `((a, d, c), f)` where `a, d, c` are the calibrated parameters
          and `f` is the value of the objective function at the minimum.
        """
        # Objective function; p = (a, d, c)
        def f(p):
            return 0.5*np.linalg.norm(
                p[0] + p[1]*(x-m)/sigma + p[2]*np.sqrt(((x-m)/sigma)**2+1) -
                w)**2

        # Gradient of the objective function
        def fprime(p):
            v1 = (x-m)/sigma
            v2 = np.sqrt(((x-m)/sigma)**2+1)
            v = p[0] + p[1]*v1 + p[2]*v2 - w
            return (np.sum(v), np.dot(v1, v), np.dot(v2, v))

        res = optimize.minimize(
            f,
            x0=(np.max(w)/2, 0, 2*sigma),
            method="SLSQP",
            jac=fprime,
            bounds=[(None, np.max(w)), (None, None), (0, 4*sigma)],
            constraints=[
                {'type': 'ineq',
                 'fun': lambda p: p[2]-p[1],
                 'jac': lambda _: (0, -1, 1)},
                {'type': 'ineq',
                 'fun': lambda p: p[2]+p[1],
                 'jac': lambda _: (0, 1, 1)},
                {'type': 'ineq',
                 'fun': lambda p: 4*sigma - p[2]-p[1],
                 'jac': lambda _: (0, -1, -1)},
                {'type': 'ineq',
                 'fun': lambda p: p[1]+4*sigma-p[2],
                 'jac': lambda _: (0, 1, -1)}])
        return res.x, res.fun

    @classmethod
    def calibrate(cls,
                  x: FloatArray,
                  w: FloatArray,
                  min_sigma: float = 1e-4,
                  max_sigma: float = 10,
                  return_minimize_result: bool = False):
        """Calibrates the parameters of the model.
        This function finds the parameters which minimize the sum of squares of
        the differences of the given total implied variances and the ones
        produced by the model.
        The two-step minimization procedure is used (by Zeliade Systems, see
        their white paper). For each pair of parameters `sigma, m`, parameters
        `a, d, c` are found by using a gradient method; then `sigma, m` are
        found by a stochastic method (namely, SLSQP and Dual Annealing are
        used).
        Args:
          x: Array of log-moneynesses
          w: Array of total implied variances.
          min_sigma, max_sigma: Bounds for `sigma` parameter.
          return_minimize_result: If True, return also the minimization result
            of `sciy.optimize.dual_annealing`.
        Returns:
          If `return_minimize_result` is True, returns a tuple `(cls, res)`,
          where `cls` is an instance of the class with the calibrated
          parameters and `res` in the optimization result returned by
          `scipy.optimize.dual_annealing`. Otherwise returns only `cls`.
        """
        res = optimize.dual_annealing(
            lambda q: cls._calibrate_adc(x, w, q[0], q[1])[1],  # q=(m, sigma)
            bounds=[(min(x), max(x)), (min_sigma, max_sigma)],
            minimizer_kwargs={"method": "nelder-mead"})
        m, sigma = res.x
        a, d, c = cls._calibrate_adc(x, w, m, sigma)[0]
        rho = d/c
        b = c/sigma
        ret = cls(a, b, rho, m, sigma)
        if return_minimize_result:
            return ret, res
        else:
            return ret
