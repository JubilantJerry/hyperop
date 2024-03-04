import torch


def compute_taylor(taylor_center, power_arange, power_scale, coefs):
    return (taylor_center ** power_arange * power_scale) @ coefs[1:] + coefs[0]


def hyperop(vals, coefs, up_fn, down_fn, up_iters, down_iters):
    assert up_iters >= 1 and down_iters >= 2
    coefs = coefs.to(device=vals.device, dtype=vals.dtype)
    power_arange = torch.arange(1, coefs.shape[0]).to(
        device=vals.device, dtype=vals.real.dtype)
    power_scale = torch.exp(-torch.lgamma(power_arange + 1))
    vals = vals + 1
    inf_mask = torch.isinf(vals) & (vals.real > 0)
    neg_mask = (vals.real < 0)
    vals_neg = vals[neg_mask]
    if vals_neg.numel() > 0:
        vals_neg = vals_neg + 1
        up_counter = torch.zeros(
            vals_neg.shape, device=vals.device, dtype=torch.long)
        for i in range(up_iters - 1):
            up_mask = (vals_neg.real < 0)
            up_counter[up_mask] += 1
            vals_neg = torch.where(up_mask, vals_neg + 1, vals_neg)
        vals = torch.masked_scatter(vals, neg_mask, vals_neg)
    high_mask = (vals.real > 1)
    vals_high = vals[high_mask]
    if vals_high.numel() > 0:
        vals_high = vals_high - 1
        down_counter = torch.zeros(
            vals_high.shape, device=vals.device, dtype=torch.long)
        for i in range(down_iters - 1):
            down_mask = (vals_high.real > 1)
            down_counter[down_mask] += 1
            vals_high = torch.where(down_mask, vals_high - 1, vals_high)
        vals = torch.masked_scatter(vals, high_mask, vals_high)
    if not torch.is_complex(vals):
        vals = torch.clip(vals, min=-1e-6, max=1 + 1e-6)
    taylor_center = (vals - 0.5).unsqueeze(-1)
    result = compute_taylor(taylor_center, power_arange, power_scale, coefs)
    if vals_neg.numel() > 0:
        result_neg = result[neg_mask]
        result_neg = down_fn(result_neg)
        for i in range(up_iters - 1):
            up_mask = (up_counter > 0)
            result_neg = torch.where(up_mask, down_fn(result_neg), result_neg)
            up_counter -= 1
        result = torch.masked_scatter(result, neg_mask, result_neg)
    if vals_high.numel() > 0:
        result_high = result[high_mask]
        result_high = up_fn(result_high)
        dhigh_mask = (down_counter > 0)
        result_dhigh = result_high[dhigh_mask]
        result_dhigh = up_fn(result_dhigh)
        down_counter_d = down_counter[dhigh_mask] - 1
        full_zero = torch.zeros_like(result_dhigh)
        for i in range(down_iters - 2):
            down_mask = (down_counter_d > 0)
            result_dhigh_in = torch.where(down_mask, result_dhigh, full_zero)
            result_dhigh = torch.where(
                down_mask, up_fn(result_dhigh_in), result_dhigh)
            down_counter_d -= 1
        result_high = torch.masked_scatter(
            result_high, dhigh_mask, result_dhigh)
        result = torch.masked_scatter(result, high_mask, result_high)
    result = torch.where(inf_mask, torch.full_like(vals, torch.inf), result)
    return result


def inv_hyperop(vals, coefs, inv_table_x, inv_table_y, up_fn, down_fn,
                up_iters, down_iters):
    assert up_iters >= 1 and down_iters >= 2
    assert not torch.is_complex(vals)
    coefs = coefs.to(device=vals.device, dtype=vals.dtype)
    power_arange = torch.arange(1, coefs.shape[0]).to(
        device=vals.device, dtype=vals.dtype)
    power_scale = torch.exp(-torch.lgamma(power_arange + 1))
    inf_mask = torch.isinf(vals) & (vals > 0)
    neg_mask = (vals < 0)
    vals_neg = vals[neg_mask]
    if vals_neg.numel() > 0:
        vals_neg = up_fn(vals_neg)
        up_counter = torch.zeros(
            vals_neg.shape, device=vals.device, dtype=torch.long)
        full_zero = torch.zeros_like(vals_neg)
        for i in range(up_iters - 1):
            up_mask = (vals_neg < 0)
            up_counter[up_mask] += 1
            vals_neg_in = torch.where(up_mask, vals_neg, full_zero)
            vals_neg = torch.where(up_mask, up_fn(vals_neg_in), vals_neg)
        vals = torch.masked_scatter(vals, neg_mask, vals_neg)
    still_neg_mask = (vals < 0)
    high_mask = (vals > 1)
    vals_high = vals[high_mask]
    if vals_high.numel() > 0:
        vals_high = down_fn(vals_high)
        down_counter = torch.zeros(
            vals_high.shape, device=vals.device, dtype=torch.long)
        for i in range(down_iters - 1):
            down_mask = (vals_high > 1)
            down_counter[down_mask] += 1
            vals_high = torch.where(down_mask, down_fn(vals_high), vals_high)
        vals = torch.masked_scatter(vals, high_mask, vals_high)
    nan_mask = torch.isnan(vals)
    vals = torch.where(nan_mask, torch.zeros_like(vals), vals)
    vals = torch.clip(vals, min=-1e-6, max=1 + 1e-6)
    x_table = inv_table_x.to(device=vals.device, dtype=vals.dtype)
    y_table = inv_table_y.to(device=vals.device, dtype=vals.dtype)
    bounds = x_table.clone()
    bounds[0] = -torch.inf
    bounds[-1] = torch.inf
    indices = torch.searchsorted(bounds, vals).flatten()
    indices = torch.cat((indices - 1, indices), dim=0)
    x = torch.index_select(x_table, 0, indices).reshape(2, *vals.shape)
    y = torch.index_select(y_table, 0, indices).reshape(2, *vals.shape)
    slope = (y[1] - y[0]) / (x[1] - x[0])
    result = slope * (vals - x[0]) + y[0]
    taylor_center = (result - 0.5).unsqueeze(-1)
    vals_back = compute_taylor(taylor_center, power_arange, power_scale, coefs)
    slope = compute_taylor(
        taylor_center, power_arange[:-1], power_scale[:-1], coefs[1:])
    result = result + (vals - vals_back) / slope
    if vals_neg.numel() > 0:
        result_neg = result[neg_mask]
        result_neg = result_neg - 1
        for i in range(up_iters - 1):
            up_mask = (up_counter > 0)
            result_neg = torch.where(up_mask, result_neg - 1, result_neg)
            up_counter -= 1
        result = torch.masked_scatter(result, neg_mask, result_neg)
    if vals_high.numel() > 0:
        result_high = result[high_mask]
        result_high = result_high + 1
        for i in range(down_iters - 1):
            down_mask = (down_counter > 0)
            result_high = torch.where(down_mask, result_high + 1, result_high)
            down_counter -= 1
        result = torch.masked_scatter(result, high_mask, result_high)
    result = result - 1
    full_nan = torch.full_like(vals, torch.nan)
    result = torch.where(nan_mask, full_nan, result)
    result = torch.where(inf_mask, torch.full_like(vals, torch.inf), result)
    result = torch.where(still_neg_mask, full_nan, result)
    return result
