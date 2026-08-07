import torch

from complextorch import integrate_spectral_rate


def test_endpoint_inclusive_spectral_integration_recovers_constant_rate():
    """Trapezoidal one-sided integration preserves a constant temporal rate."""
    sampling_frequency = 200.0
    frequencies = torch.linspace(0.0, sampling_frequency / 2.0, 1001, dtype=torch.float64)
    values = torch.full((3, frequencies.numel()), 2.75, dtype=torch.float64)

    integrated = integrate_spectral_rate(
        values, frequencies, sampling_frequency=sampling_frequency
    )

    torch.testing.assert_close(integrated, torch.full((3,), 2.75, dtype=torch.float64))


def test_faes_half_open_integration_is_exact_bin_mean():
    """Faes/HOP half-open FFT grids use the exact arithmetic-bin mean."""
    sampling_frequency = 100.0
    nfft = 512
    frequencies = (
        torch.arange(nfft, dtype=torch.float64)
        * sampling_frequency
        / (2.0 * nfft)
    )
    values = torch.arange(nfft, dtype=torch.float64).reshape(1, -1)

    integrated = integrate_spectral_rate(
        values,
        frequencies,
        sampling_frequency=sampling_frequency,
        half_open=True,
    )

    torch.testing.assert_close(integrated, values.mean(dim=-1))


def test_faes_half_open_integration_rejects_nonuniform_grid():
    """Half-open mode rejects grids that do not match the Faes FFT convention."""
    frequencies = torch.tensor([0.0, 0.1, 0.21, 0.3], dtype=torch.float64)
    values = torch.ones(4, dtype=torch.float64)

    try:
        integrate_spectral_rate(values, frequencies, half_open=True)
    except ValueError as exc:
        assert "uniform" in str(exc) or "spacing" in str(exc)
    else:
        raise AssertionError("nonuniform half-open grid should be rejected")
