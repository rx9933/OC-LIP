import numpy as np
import matplotlib.pyplot as plt

def fourier_frequencies(Ty, Nf):
    """
    Ty = (t1, tf): time window for the sensor path
    Nf: number of Fourier modes

    Returns omegas (Nf,) where omega_k = 2*pi*k / (tf - t1)
    """
    t1, tf = Ty
    T = tf - t1
    if T <= 0:
        raise ValueError("Ty must satisfy tf > t1.")
    ks = np.arange(1, Nf + 1, dtype=float)
    return 2.0 * np.pi * ks / T


def fourier_path(t, xbar, coeffs, omegas, Ty=None):
    """
    2D Fourier path:
      c(t) = xbar + sum_{k=1..Nf} [ a_k cos(omega_k * tau) + b_k sin(omega_k * tau) ] in x
                    + sum_{k=1..Nf} [ c_k cos(omega_k * tau) + d_k sin(omega_k * tau) ] in y

    Inputs:
      t: scalar time
      xbar: (2,) base point
      coeffs: (Nf, 4) with columns [a_k, b_k, c_k, d_k]
      omegas: (Nf,)
      Ty (optional): (t1, tf). If provided, we shift time so tau = t - t1.
                     (This makes the series "start" nicely at t1.)

    Returns:
      (2,) numpy array [x(t), y(t)]
    """
    xbar = np.asarray(xbar, dtype=float).reshape(2,)
    coeffs = np.asarray(coeffs, dtype=float)
    omegas = np.asarray(omegas, dtype=float)

    if coeffs.shape != (len(omegas), 4):
        raise ValueError(f"coeffs must be shape (Nf, 4). Got {coeffs.shape}, expected ({len(omegas)}, 4).")

    if Ty is None:
        tau = float(t)
    else:
        t1, tf = Ty
        tau = float(t) - float(t1)

    a = coeffs[:, 0]; b = coeffs[:, 1]
    c = coeffs[:, 2]; d = coeffs[:, 3]

    cosw = np.cos(omegas * tau)
    sinw = np.sin(omegas * tau)

    dx = np.sum(a * cosw + b * sinw)
    dy = np.sum(c * cosw + d * sinw)

    return xbar + np.array([dx, dy], dtype=float)


def fourier_scalar(t, cossin_coeffs, omegas):
    """
    cossin_coeffs: array shape (Nf, 2) where [:,0]=cos coeffs, [:,1]=sin coeffs
    returns sum_j a_j cos(ω_j t) + b_j sin(ω_j t)
    """
    val = 0.0
    for j in range(len(omegas)):
        a = cossin_coeffs[j, 0]
        b = cossin_coeffs[j, 1]
        w = omegas[j]
        val += a * np.cos(w*t) + b * np.sin(w*t)
    return val

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def unicycle_fourier_controls(t, omegas, coeffs_v, coeffs_w, v0=0.15, w0=0.0):
    """
    v(t) = v0 + Fourier(t; coeffs_v)
    ω(t) = w0 + Fourier(t; coeffs_w)
    """
    v = v0 + fourier_scalar(t, coeffs_v, omegas)
    w = w0 + fourier_scalar(t, coeffs_w, omegas)
    return v, w

def integrate_unicycle_path(times, x0, theta0, omegas, coeffs_v, coeffs_w,
                            v0=0.15, w0=0.0, eps=1e-3, substeps=10):
    """
    Option B: Integrate unicycle dynamics with Fourier controls.
    Returns positions c(t_n) mapped smoothly into (eps, 1-eps)^2.
    """
    x = np.array(x0, dtype=float)
    theta = float(theta0)
    pts = []

    for i in range(len(times)):
        if i == 0:
            # record initial
            raw = x.copy()
        else:
            t0, t1 = times[i-1], times[i]
            h = (t1 - t0) / substeps
            t = t0
            for _ in range(substeps):
                v, w = unicycle_fourier_controls(t, omegas, coeffs_v, coeffs_w, v0=v0, w0=w0)

                # optional: keep v nonnegative smoothly (so you don't go backwards unintentionally)
                v = 1e-6 + np.log1p(np.exp(v))  # softplus

                # integrate
                x[0] += h * v * np.cos(theta)
                x[1] += h * v * np.sin(theta)
                theta += h * w
                t += h

            raw = x.copy()

        # smooth feasibility map
        mapped = eps + (1.0 - 2.0*eps) * sigmoid(raw)
        pts.append(mapped)

    return np.array(pts)
def plot_targets(targets, title = 'targets.pdf'):
    plt.figure(figsize=(6, 6))

    plt.scatter(targets[:, 0], targets[:, 1],
                c='red', s=40, label='Sensor locations')

    plt.plot(targets[:, 0], targets[:, 1],
            'r--', linewidth=1, alpha=0.7)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.gca().set_aspect('equal', 'box')
    plt.grid(True, alpha=0.3)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fixed sensor locations sampled along Fourier path')
    plt.legend()
    # plt.show()
    plt.save_fig(title)
def plot_eigenvalues(lmbda, k, Vh, V):
    plt.figure()
    plt.plot(range(k), lmbda, 'b*', range(k+1), np.ones(k+1), '-r')
    plt.yscale('log')
    plt.xlabel('index')
    plt.ylabel('eigenvalue')
    plt.title('Hessian eigenvalue decay')
    plt.show()

    nb.plot_eigenvectors(
        Vh, V,
        mytitle="Posterior Hessian Eigenvectors",
        which=[0, 1, 2, 5, 10, 20, 30, 45, 60]
    )
    plt.show()
