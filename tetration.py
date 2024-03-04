import torch
import hyperop


COEFS = torch.tensor([
     4.98563288e-01,  9.56755190e-01,  2.14095880e-02,  9.78884449e-01, # noqa
    -4.08072599e-01,  4.40161440e+00, -7.30564963e+00,  4.84161257e+01, # noqa
    -1.66089982e+02,  1.42583591e+03, -8.03934321e+03,  8.62533112e+00, # noqa
    -4.43738314e+01,  2.38188324e-02, -1.12655831e-01,  4.03379964e-05, # noqa
    -1.76528447e-04,  4.74344210e-08, -1.93129951e-07,  4.15408303e-11, # noqa
    -1.58109197e-10,  2.83568443e-14, -1.01315841e-13,  1.55843111e-17, # noqa
    -5.24617791e-17], # noqa
    dtype=torch.float64)


def tetrate(vals):
    return hyperop.hyperop(vals, COEFS, torch.exp, torch.log, 1, 4)


INV_TABLE_Y = torch.linspace(0, 1, 5001, dtype=torch.float64)
INV_TABLE_X = tetrate(INV_TABLE_Y - 1)


def inv_tetrate(vals):
    return hyperop.inv_hyperop(
        vals, COEFS, INV_TABLE_X, INV_TABLE_Y, torch.exp, torch.log, 1, 4)
