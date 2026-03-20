def fourier_frequencies(Ty, Nf):
    """Fourier angular frequencies for time window Ty = (t_start, t_end)."""
    T = Ty[1] - Ty[0]
    return np.array([2.0 * np.pi * (k + 1) / T for k in range(Nf)])


def fourier_path(t, xbar, coeffs, omegas):
    """
    Evaluate the Fourier curve at parameter t.

    Parameters
    ----------
    t      : float
    xbar   : (2,) mean position [x̄, ȳ]
    coeffs : (Nf, 4) array  [θ_k, φ_k, ψ_k, η_k] per mode
    omegas : (Nf,)  angular frequencies

    Returns
    -------
    (2,) array  [x(t), y(t)]
    """
    x = xbar[0]
    y = xbar[1]
    for k, w in enumerate(omegas):
        x += coeffs[k, 0] * np.cos(w * t) + coeffs[k, 1] * np.sin(w * t)
        y += coeffs[k, 2] * np.cos(w * t) + coeffs[k, 3] * np.sin(w * t)
    return np.array([x, y])


def m_to_xbar_coeffs(m_fourier, K):
    """Unpack flat optimisation vector → (xbar, coeffs)."""
    xbar   = m_fourier[:2].copy()
    coeffs = np.zeros((K, 4))
    for k in range(K):
        coeffs[k, 0] = m_fourier[2 + 4*k]      # θ_k
        coeffs[k, 1] = m_fourier[3 + 4*k]      # φ_k
        coeffs[k, 2] = m_fourier[4 + 4*k]      # ψ_k
        coeffs[k, 3] = m_fourier[5 + 4*k]      # η_k
    return xbar, coeffs


def xbar_coeffs_to_m(xbar, coeffs, K):
    """Pack (xbar, coeffs) → flat optimisation vector."""
    m = np.zeros(4*K + 2)
    m[:2] = xbar
    for k in range(K):
        m[2 + 4*k] = coeffs[k, 0]
        m[3 + 4*k] = coeffs[k, 1]
        m[4 + 4*k] = coeffs[k, 2]
        m[5 + 4*k] = coeffs[k, 3]
    return m