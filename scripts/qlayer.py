import torch
import torch.nn as nn
import torch.nn.functional as F

import lowbit_pack as L

G = L.GROUP
EXPORT_TERN5 = False  # True: export() writes ternary (kind 2) layers as kind 5, 5 trits per byte


def init_scale(w, kind):
    """per-group scale minimizing the quantization MSE of w [out,in] (fp32) -> [out, in/G] fp32"""
    out, n = w.shape
    a = w.reshape(out, n // G, G).abs()
    if kind == 1:  # binary: alpha = mean|w|, levels +-0.5*scale
        return 2 * a.mean(-1)
    if kind == 2:  # ternary: exact optimum over the support size k
        srt = a.sort(-1, descending=True).values
        cs = srt.cumsum(-1)
        k = torch.arange(1, G + 1, device=w.device, dtype=w.dtype)
        kbest = (cs ** 2 / k).argmax(-1, keepdim=True)
        return cs.gather(-1, kbest).squeeze(-1) / (kbest.squeeze(-1) + 1)
    _, bits, off = L.KINDS[kind]
    qmax = 2 ** bits - 1
    wg = w.reshape(out, n // G, G)
    base = a.amax(-1) / (qmax / 2)
    best_err = best_s = None
    for f in torch.linspace(0.4, 1.05, 66, device=w.device):
        s = (base * f).clamp_min(1e-9)
        q = torch.round(wg / s.unsqueeze(-1) + off).clamp_(0, qmax) - off
        err = ((q * s.unsqueeze(-1) - wg) ** 2).sum(-1)
        if best_err is None:
            best_err, best_s = err, s
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_s = torch.where(better, s, best_s)
    return best_s


def fake_quant(w, s, kind):
    """straight-through fake quantization. w [out,in] fp32, s [out,in/G] fp32 > 0"""
    out, n = w.shape
    x = w.reshape(out, n // G, G) / s.unsqueeze(-1)
    _, bits, off = L.KINDS[kind]
    c = (x + off).clamp(0, 2 ** bits - 1 if kind != 2 else 2)
    if kind == 1:
        cq = c + ((x > 0).to(c.dtype) - c).detach()
    else:
        cq = c + (torch.round(c) - c).detach()
    return ((cq - off) * s.unsqueeze(-1)).reshape(out, n)


class QLinear(nn.Module):
    def __init__(self, weight, kind):
        super().__init__()
        self.kind = kind
        w = weight.detach().float()
        self.w = nn.Parameter(w.clone())
        self.log_s = nn.Parameter(init_scale(w, kind).clamp_min(1e-8).log())
        self.register_parameter("lora_a", None)
        self.register_parameter("lora_b", None)

    @property
    def weight(self):  # comfy.ops.linear_input_act probes linear.weight
        return None

    def scale(self):
        return self.log_s.exp()

    def quantized_weight(self):
        return fake_quant(self.w, self.scale(), self.kind)

    cached_w = None  # the trainer sets the bf16 fake-quantized weight once per optimizer step

    def forward(self, x):
        w = self.cached_w if self.cached_w is not None else self.quantized_weight().to(x.dtype)
        out = F.linear(x, w)
        if self.lora_a is not None:
            out = out + F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))
        return out

    @torch.no_grad()
    def init_lora(self, rank, cov):
        """data-aware low-rank correction: min ||(W - Q - B A) S^(1/2)||_F with S = E[x x^T] (cov [in,in] fp32)"""
        E = self.w.float() - self.quantized_weight()
        lam = 1e-3 * cov.diagonal().mean()
        ev, V = torch.linalg.eigh(cov + lam * torch.eye(cov.shape[0], device=cov.device))
        ev = ev.clamp_min(lam)
        sq, isq = (V * ev.sqrt()) @ V.T, (V * ev.rsqrt()) @ V.T
        U, S, Vs = torch.svd_lowrank(E @ sq, q=rank + 8, niter=3)
        U, S, Vs = U[:, :rank], S[:rank], Vs[:, :rank]
        self.lora_b = nn.Parameter(U * S.sqrt())
        self.lora_a = nn.Parameter((S.sqrt().unsqueeze(1) * Vs.T) @ isq)

    def export(self):
        """-> (packed uint8, scale fp16, kind uint8) in moss_lowbit_v1"""
        kind = 5 if self.kind == 2 and EXPORT_TERN5 else self.kind
        sc = self.scale().detach().half()
        codes = L.quantize_to_codes(self.w.detach(), sc, self.kind)
        return L.pack_codes(codes, L.KINDS[kind][1]).cpu(), sc.cpu(), torch.tensor(kind, dtype=torch.uint8)
