import torch
import hyperop
import hexation


COEFS = torch.tensor([
     4.92740452e-01, 9.86246306e-01, 3.97193686e-02, 3.12337587e-01, # noqa
     8.49149119e-01, 1.30090207e+00, 3.50012813e+00, 1.92206727e+01, # noqa
     7.49945359e+01, 2.31106228e+02, 2.94065994e+02, 1.39780447e+00, # noqa
     1.62197231e+00, 3.85982226e-03, 4.11687432e-03], # noqa
    dtype=torch.float64)


def heptate(vals):
    return hyperop.hyperop(
        vals, COEFS, hexation.hexate, hexation.inv_hexate, 20, 2)


INV_TABLE_Y = torch.linspace(0, 1, 5001, dtype=torch.float64)
INV_TABLE_X = heptate(INV_TABLE_Y - 1)


def inv_heptate(vals):
    return hyperop.inv_hyperop(
        vals, COEFS, INV_TABLE_X, INV_TABLE_Y,
        hexation.hexate, hexation.inv_hexate, 20, 2)
