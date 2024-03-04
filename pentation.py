import torch
import hyperop
import tetration


COEFS = torch.tensor([
     4.91105155e-01,  9.78846550e-01,  5.17367476e-02,  4.98655043e-01, # noqa
     8.89264807e-01,  6.58577629e-01,  4.85212090e+00,  9.96299136e+00, # noqa
     7.41160600e+01,  2.09790987e+02, -1.93233743e+03,  1.26903442e+00, # noqa
    -1.06676014e+01,  3.50438285e-03, -2.70844068e-02], # noqa
    dtype=torch.float64)


def pentate(vals):
    return hyperop.hyperop(
        vals, COEFS, tetration.tetrate, tetration.inv_tetrate, 20, 3)


INV_TABLE_Y = torch.linspace(0, 1, 5001, dtype=torch.float64)
INV_TABLE_X = pentate(INV_TABLE_Y - 1)


def inv_pentate(vals):
    return hyperop.inv_hyperop(
        vals, COEFS, INV_TABLE_X, INV_TABLE_Y,
        tetration.tetrate, tetration.inv_tetrate, 20, 3)
