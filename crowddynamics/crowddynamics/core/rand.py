import numba
import numpy as np
import scipy.stats

from numba import f8, i8


def truncnorm(start, end, loc=0.0, scale=1.0, abs_scale=None, size=1,
              random_state=None):
    """Truncated normal distribution from ``scipy.stats``.

    Args:
        start (float):
        end (float):
        loc (float):
        scale (float|numpy.ndarray):
        abs_scale: Absolute scale ``scale = abs_scale / max(abs(start), abs(end))
        size (int):
        random_state (int, optional):

    Returns:
        numpy.ndarray:

    References:
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        - https://en.wikipedia.org/wiki/Truncated_normal_distribution
    """
    _scale = abs_scale / max(abs(start), abs(end)) if abs_scale else scale
    #np.random.seed(0)
    tn = scipy.stats.truncnorm.rvs(
        start, end, loc=loc, scale=_scale, size=size, random_state=random_state
    )
    return tn


def random_vector(size, orient=(0.0, 2.0 * np.pi), mag=1.0):
    #np.random.seed(0)
    orientation = np.random.uniform(orient[0], orient[1], size=size)
    return mag * np.stack((np.cos(orientation), np.sin(orientation)), axis=1)


@numba.jit((f8, f8), nopython=True, nogil=True)
def poisson_clock(interval, dt):
    r"""Generated points in time using Poisson process.

    Time between updates are independent random variables defined

    .. math::

       \tau_i \sim \operatorname{Exp}(\lambda)

    Where the ``rate`` parameter is

    .. math::

       \operatorname{E}(\tau) = \frac{1}{\lambda} = interval

    Times when agent updates its strategy

    .. math::

       \begin{split}\begin{cases}
       T_n = \tau_1 + \ldots + \tau_n, & n \geq 1 \\
       T_0 = 0
       \end{cases}\end{split}

    Sequence of update times

    .. math::

       \{T_1, T_2, T_3, \ldots, T_n\}, \quad T_n < \Delta t

    Number of arrivals by the :math:`s`

    .. math::

       N(s) &= \max\{n : T_n \leq s\} \\
       N(s) &\sim \operatorname{Poi}(\lambda)

    Args:
        interval (float):
            Expected frequency of update.

        dt (float):
            Discrete timestep :math:`\Delta t` aka time window.

    Yields:
        float:
            Moments in the time window when the strategies should be updated.

    """
    # Numpy exponential distribution's scale parameter is equal to
    # 1/lambda which is why we can supply interval directly into the
    # np.random.exponential.
    t_tot = np.random.exponential(scale=interval)
    while t_tot < dt:
        yield t_tot
        t_tot += np.random.exponential(scale=interval)


@numba.jit((i8[:], f8, f8), nopython=True, nogil=True)
def poisson_timings(players, interval, dt):
    r"""
    Update times for all agent in the game using Poisson clock.

    Args:
        players (numpy.ndarray):
            Indices of the players.

        interval (float):
        dt (float):

    Returns:
        list: List of indices of agents sorted by their update times.

    """
    # Compute update times for all agents.
    times = []
    indices = []
    # TODO: check shuffle doesn't cause any side effects
    np.random.shuffle(players)
    for i in players:
        for t in poisson_clock(interval, dt):
            times.append(t)
            indices.append(i)

    # Sort the indices by the update times
    # noinspection PyTypeChecker
    for j in np.argsort(np.array(times)):
        yield indices[j]


def estimate_number_poisson_timings(interval, dt):
    """Estimated number of points generated by poisson process.

    .. math::
       N &\sim \operatorname{Poi}(\lambda)

    Solve for number of points :math:`n`

    .. math::
       \Pr(N \leq n) = p

    """
    pass
