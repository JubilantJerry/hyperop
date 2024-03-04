import torch
import hyperop
import pentation


COEFS = torch.tensor([
     4.91532626e-01,  9.83715391e-01,  4.81631136e-02,  3.76428320e-01, # noqa
     9.09639935e-01,  1.03104425e+00,  3.27687636e+00,  1.98164733e+01, # noqa
     7.37194546e+01,  1.53965509e+02, -4.59712672e+02,  9.31090898e-01, # noqa
    -2.53881099e+00,  2.57093396e-03, -6.44668172e-03], # noqa
    dtype=torch.float64)


def hexate(vals):
    return hyperop.hyperop(
        vals, COEFS, pentation.pentate, pentation.inv_pentate, 3, 2)


INV_TABLE_Y = torch.linspace(0, 1, 5001, dtype=torch.float64)
INV_TABLE_X = hexate(INV_TABLE_Y - 1)


def inv_hexate(vals):
    return hyperop.inv_hyperop(
        vals, COEFS, INV_TABLE_X, INV_TABLE_Y,
        pentation.pentate, pentation.inv_pentate, 3, 2)
