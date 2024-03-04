import numpy as np
import torch
# import tetration
# import pentation
# import hexation
# import heptation

up_fn = torch.exp
down_fn = torch.log
# up_fn = tetration.tetrate
# down_fn = tetration.inv_tetrate
# up_fn = pentation.pentate
# down_fn = pentation.inv_pentate
# up_fn = hexation.hexate
# down_fn = hexation.inv_hexate
# up_fn = heptation.heptate
# down_fn = heptation.inv_heptate

points1 = torch.linspace(0, 1, 5001, dtype=torch.complex128)
points1 = torch.cat((points1, points1 + 0.5j), dim=0)  # For tetration
# points1 = torch.cat((points1, points1 + 0.01j), dim=0)  # For others
points2 = up_fn(points1)

power_arange = torch.arange(1, 25).to(dtype=torch.float64)  # For tetration
# power_arange = torch.arange(1, 15).to(dtype=torch.float64)  # For others
power_scale = torch.exp(-torch.lgamma(power_arange + 1))
powers1 = (points1[:, None] - 1) ** power_arange * power_scale
powers2 = (points2[:, None] - 1) ** power_arange * power_scale
# val_at_origin = 0.498563287966823  # For tetration, half-iteration
val_at_origin = 0.0108855715833891  # For tetration
# val_at_origin = 0.00996923169961  # For pentation
# val_at_origin = 0.0098968614575  # For hexation
# val_at_origin = 0.0099037649  # For heptation
# val_at_origin = 0.00991905223  # For octation

coefs = torch.zeros(1 + power_arange.shape[0], dtype=torch.float64)
coefs[0] = val_at_origin + 1
coefs[1] = 1
powers3 = ((-1) ** power_arange * power_scale).unsqueeze(0)
eye = torch.eye(coefs.shape[0], dtype=torch.float64)


def mean_sqnorm(vals):
    return (vals @ vals.conj()).real / vals.shape[0]


def compute_taylor(powers, coefs):
    return (powers @ coefs[1:]) + coefs[0]


def get_loss(coefs):
    coefs_complex = coefs.to(dtype=torch.complex128)
    loss1 = mean_sqnorm(
        up_fn(compute_taylor(powers1, coefs_complex)) -
        compute_taylor(powers2, coefs_complex))
    loss2 = mean_sqnorm(compute_taylor(powers3, coefs) - val_at_origin)
    return loss1 + loss2


curr_loss = get_loss(coefs)
for i in range(1000):
    hess = torch.autograd.functional.hessian(get_loss, coefs)
    grad = torch.autograd.functional.jacobian(get_loss, coefs)
    mult = 1.0
    if torch.isnan(hess).sum().item() > 0:
        print("Intervening...")
        coefs = coefs - 0.001 * grad
        continue
    eigvals, eigvecs = torch.linalg.eigh(hess)
    eigvals = torch.abs(eigvals)
    direction = eigvecs @ ((eigvecs.T @ grad) / eigvals)
    done = False
    while True:
        new_loss = get_loss(coefs - mult * direction)
        if new_loss < curr_loss or mult < 1e-30:
            if mult < 1e-10:
                done = True
            break
        mult *= 0.5
    if done:
        print(f"Done on iteration {i}")
        break
    coefs = coefs - mult * direction
    curr_loss = new_loss
    print(curr_loss.item())


def make_small_step(v):
    if v < 0:
        return down_fn(make_small_step(up_fn(v)))
    if v > np.e:
        return up_fn(make_small_step(down_fn(v)))
    return compute_taylor((v - 1) ** power_arange * power_scale, coefs)


v = torch.tensor(0.0, dtype=torch.float64)
y_vals = [0]
for i in range(1, 1000):
    v = make_small_step(v)
    v_item = v.item()
    print(f"{i}: {v_item}")
    y_vals.append(v_item)
    if v > (1 - 1e-4):
        break
y_vals = np.array(y_vals)
x_vals = np.linspace(0, 1, y_vals.shape[0])
power_arange = power_arange.numpy()
power_scale = power_scale.numpy()
powers4 = (x_vals[:, None] - 0.5) ** power_arange * power_scale
powers4 = np.concatenate((np.ones_like(powers4[:, :1]), powers4), axis=1)
iter_coefs = np.linalg.lstsq(powers4, y_vals, rcond=None)[0]
print(repr(iter_coefs))


while True:
    v = input("Enter value: ").strip()
    try:
        print(make_small_step(
            torch.tensor(float(v), dtype=torch.float64)).item())
    except ValueError as v:
        print(v)
