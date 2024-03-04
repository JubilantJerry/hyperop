import torch
import hyperop
import heptation


COEFS = torch.tensor([
     4.93874330e-01, 9.88086580e-01, 3.16105886e-02, 2.65996117e-01, # noqa
     8.01832720e-01, 1.47902289e+00, 3.60055867e+00, 1.84107886e+01, # noqa
     8.20436360e+01, 2.64616319e+02, 5.80285244e+02, 1.60055697e+00, # noqa
     3.20176459e+00, 4.41975910e-03, 8.12761884e-03], # noqa
    dtype=torch.float64)


def octate(vals):
    return hyperop.hyperop(
        vals, COEFS, heptation.heptate, heptation.inv_heptate, 5, 2)


INV_TABLE_Y = torch.linspace(0, 1, 5001, dtype=torch.float64)
INV_TABLE_X = octate(INV_TABLE_Y - 1)


def inv_octate(vals):
    return hyperop.inv_hyperop(
        vals, COEFS, INV_TABLE_X, INV_TABLE_Y,
        heptation.heptate, heptation.inv_heptate, 5, 2)
