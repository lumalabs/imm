import torch
from torch.nn.parallel import DistributedDataParallel


class IMMLoss(torch.nn.Module):
    """Inductive Moment Matching (IMM) Loss
    https://arxiv.org/abs/2503.07565

    Defaults for CIFAR-10 (Table 5)
    """

    def __init__(
        self,
        M=4,  # group size (must be divisible by batch size)
        a=1,  # a = 1 one‑step, a=2 multi‑step/FP16
        b=5,  # shift sigmoid
        k=15,  # eta r mapping fn power
        eta_max=160,  # eta r mapping fn max
        eta_min=0,  # eta r mapping fn min
        eps=1e-8,
    ):
        super().__init__()
        self.M = M
        self.a = a
        self.b = b
        self.k = k
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.eps = eps

    def loss_weight(self, net, t):
        lamb = net.get_logsnr(t)
        # negative time derivative of lamb
        if net.noise_schedule == "vp_cosine":
            dlamb_dt = 2 * torch.pi / (torch.sin(torch.pi * t) + self.eps)
        if net.noise_schedule == "fm":
            dlamb_dt = 2.0 / (t * (1 - t) + self.eps)
        alpha_t, sigma_t = net.get_alpha_sigma(t)
        alpha_t = alpha_t.detach()
        sigma_t = sigma_t.detach()

        weight = (
            0.5
            * torch.sigmoid(self.b - lamb)
            * dlamb_dt
            * (alpha_t**self.a / (alpha_t**2 + sigma_t**2 + self.eps))
        )
        return weight

    def kernel_weight(self, net, t, s):
        if net.f_type == "identity":
            c_out = torch.ones_like(t)
        elif net.f_type == "simple_edm":
            alpha_t, sigma_t = net.get_alpha_sigma(t)
            alpha_s, sigma_s = net.get_alpha_sigma(s)
            c_out = (
                -(alpha_s * sigma_t - alpha_t * sigma_s)
                * (alpha_t**2 + sigma_t**2).rsqrt()
                * net.sigma_data
            )
        elif net.f_type == "euler_fm":
            c_out = -t * net.sigma_data

        weight = 1 / torch.abs(c_out)
        return weight[:, None, None]  # [G, 1, 1]

    def compute_r(self, net, t, s):
        alpha_t, sigma_t = net.get_alpha_sigma(t)

        eta_t = sigma_t / alpha_t
        eps = (self.eta_max - self.eta_min) / 2**self.k
        eta = eta_t - eps

        if net.noise_schedule == "vp_cosine":  # inverse of tan(pi t/2)
            r = (2.0 / torch.pi) * torch.atan(eta)
        elif net.noise_schedule == "fm":  # inverse of t/(1-t)
            r = eta / (1.0 + eta)

        return torch.maximum(s, r)

    def forward(self, model, x, class_labels=None):
        net = model.module if isinstance(model, DistributedDataParallel) else model

        B = x.shape[0]
        M = self.M
        assert B % M == 0, f"Batch size ({B}) must be divisible by M ({M})"
        G = B // M

        # sample one (t,s,r) per group
        t_g = torch.rand(G, device=x.device) * (net.T - net.eps) + net.eps
        s_g = torch.rand(G, device=x.device) * (t_g - net.eps) + net.eps
        r_g = self.compute_r(net, t_g, s_g)

        # repeat to per-sample vectors
        t = t_g.repeat_interleave(M).view(B, 1, 1, 1)
        s = s_g.repeat_interleave(M).view(B, 1, 1, 1)
        r = r_g.repeat_interleave(M).view(B, 1, 1, 1)

        noise = torch.randn_like(x) * net.sigma_data
        x_t = net.ddim(noise, x, t=torch.ones_like(t), s=t)
        x_r = net.ddim(x_t, x, t=t, s=r)

        y_t = model(x_t, t, s, class_labels=class_labels).view(G, M, -1)  # [G, M, D]
        with torch.no_grad():  # stop grad
            y_r = model(x_r, r, s, class_labels=class_labels).view(
                G, M, -1
            )  # [G, M, D]

        # group‑wise loss and kernel weights
        w_l = self.loss_weight(net, t_g)  # [G]
        w_k = self.kernel_weight(net, t_g, s_g)  # [G]

        # pairwise distances & Laplace kernels [G, M, M]
        dist_tt = torch.cdist(y_t, y_t, p=2).clamp(min=self.eps)
        dist_rr = torch.cdist(y_r, y_r, p=2).clamp(min=self.eps)
        dist_tr = torch.cdist(y_t, y_r, p=2).clamp(min=self.eps)

        D = y_t.shape[-1]
        K_tt = torch.exp(-w_k * dist_tt / D)
        K_rr = torch.exp(-w_k * dist_rr / D)
        K_tr = torch.exp(-w_k * dist_tr / D)

        # v‑statistic MMD per group
        mmd_g = (K_tt + K_rr - 2 * K_tr).mean(dim=(1, 2))  # [G]
        return torch.mean(w_l * mmd_g)
