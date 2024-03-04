from matplotlib import pyplot as plt
import torch
import tetration
import pentation
import hexation
import heptation
import octation

x_vals = torch.linspace(10.0, 1000.0, 1000, dtype=torch.float64)
plt.figure(figsize=(16, 9))
plt.plot(x_vals, hexation.hexate(hexation.inv_hexate(x_vals) - 0.5))
plt.title("inv_pentate(half_pentate(x))")
plt.tight_layout()
plt.show()

x_vals = torch.linspace(0.0, 13.0, 1000, dtype=torch.float64)
plt.figure(figsize=(16, 9))
plt.plot(x_vals, pentation.pentate(hexation.hexate(
    hexation.inv_hexate(x_vals) - 0.5) - 1))
plt.title("inv_tetrate(half_pentate(x))")
plt.tight_layout()
plt.show()

x_vals = torch.linspace(0.0, 4.0, 1000, dtype=torch.float64)
plt.figure(figsize=(16, 9))
plt.plot(x_vals, tetration.tetrate(pentation.pentate(hexation.hexate(
    hexation.inv_hexate(x_vals) - 0.5) - 1) - 1))
plt.title("inv_trinate(half_pentate(x))")
plt.tight_layout()
plt.show()

x_vals = torch.linspace(0.0, 1.1, 1000, dtype=torch.float64)
plt.figure(figsize=(16, 9))
plt.plot(x_vals, pentation.pentate(hexation.hexate(
    hexation.inv_hexate(x_vals) + 0.5)))
plt.title("half_pentate(x)")
plt.tight_layout()
plt.show()

x_vals = torch.linspace(-6.5, 4.0, 1000, dtype=torch.float64)
plt.figure(figsize=(16, 9))
plt.plot(x_vals, torch.e + x_vals, label='unation')
plt.plot(x_vals, torch.e * x_vals, label='duation')
plt.plot(x_vals, torch.exp(x_vals), label='trination')
plt.plot(x_vals, tetration.tetrate(x_vals), label='tetration')
plt.plot(x_vals, pentation.pentate(x_vals), label='pentation')
plt.plot(x_vals, hexation.hexate(x_vals), label='hexation')
plt.plot(x_vals, heptation.heptate(x_vals), label='heptation')
plt.plot(x_vals, octation.octate(x_vals), label='octation')
plt.xlim([-6.5, 4.0])
plt.ylim([-10, 30])
plt.legend()
plt.tight_layout()
plt.show()

x_vals = torch.linspace(-20.0, 20.0, 1000, dtype=torch.float64)
plt.figure(figsize=(16, 9))
plt.plot(x_vals, x_vals - torch.e, label='inv_unation')
plt.plot(x_vals, x_vals / torch.e, label='inv_duation')
plt.plot(x_vals, torch.log(x_vals), label='inv_trination')
plt.plot(x_vals, tetration.inv_tetrate(x_vals), label='inv_tetration')
plt.plot(x_vals, pentation.inv_pentate(x_vals), label='inv_pentation')
plt.plot(x_vals, hexation.inv_hexate(x_vals), label='inv_hexation')
plt.plot(x_vals, heptation.inv_heptate(x_vals), label='inv_heptation')
plt.plot(x_vals, octation.inv_octate(x_vals), label='inv_octation')
plt.xlim([-20.0, 20.0])
plt.ylim([-6.5, 4.0])
plt.legend()
plt.tight_layout()
plt.show()
