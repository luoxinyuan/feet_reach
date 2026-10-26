import torch
from active_adaptation.utils.jacobian_guidance import allocate_jacobian_forces, reconstruction


def test_known_wrench_and_virtual_work():
    torch.manual_seed(7)
    j = torch.randn(12, 3, 3, 35, dtype=torch.float64)
    f = torch.randn(12, 3, 3, dtype=torch.float64)
    qd = torch.randn(12, 35, dtype=torch.float64)
    target = torch.einsum('nkcd,nkc->nd', j, f)
    weights = torch.ones(35, dtype=torch.float64)
    actual = allocate_jacobian_forces(j, target, weights, torch.ones(12, 3, dtype=torch.bool),
                                      torch.zeros_like(f), 1e-10, 0.)
    torch.testing.assert_close(actual, f)
    assert reconstruction(j, actual, target, weights)[2].max() < 1e-8
    torch.testing.assert_close((target * qd).sum(-1), (f * torch.einsum('nkcd,nd->nkc', j, qd)).sum((1, 2)))


def test_mask_degenerate_and_smoothing():
    j = torch.zeros(2, 3, 3, 35)
    active = torch.tensor([[True, False, False], [False, True, True]])
    previous = torch.ones(2, 3, 3)
    f = allocate_jacobian_forces(j, torch.ones(2, 35), torch.ones(35), active, previous)
    torch.testing.assert_close(f, .5 * previous * active[..., None])
    assert torch.isfinite(f).all()


def test_com_to_point_shift_sign():
    from active_adaptation.utils.jacobian_guidance import point_linear_jacobian
    raw = torch.eye(6).reshape(1, 1, 6, 6)
    offset = torch.tensor([[[1., 0., 0.]]])
    jac = point_linear_jacobian(raw, offset)
    velocity = torch.tensor([[0., 0., 0., 0., 0., 2.]])
    torch.testing.assert_close(torch.einsum('nkcd,nd->nkc', jac, velocity), torch.tensor([[[0., 2., 0.]]]))
    f = torch.tensor([[[0., 3., 0.]]])
    torch.testing.assert_close(torch.einsum('nkcd,nkc->nd', jac, f), torch.tensor([[0., 3., 0., 0., 0., 3.]]))
