import math
import os
import sys
import torch
from torch.utils.data import DataLoader
from importlib.util import spec_from_file_location, module_from_spec

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)


def load(alias, filename):
    spec = spec_from_file_location(alias, os.path.join(BASE, filename))
    mod = module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


cfg = load("cfg_mod", "config.py")
prep = load("prep_mod", "01_data_preprocessing.py")
ode_mod = load("ode_mod", "02_physics_latent.py")
enc_mod = load("enc_mod", "03_model_encoder.py")
dec_mod = load("dec_mod", "04_model_decoder.py")
loss_mod = load("loss_mod", "07_losses.py")
train_mod = load("train_mod", "08_training.py")

for name in [
    "reconstruction_loss", "ode_residual_loss", "bounds_loss",
    "monotonicity_loss", "temperature_ordering_loss",
    "multistep_prediction_loss", "adversarial_generator_loss",
    "adversarial_discriminator_loss", "distribution_matching_loss",
    "total_physics_loss",
]:
    setattr(train_mod, name, getattr(loss_mod, name))


dataset = prep.load_dataset(cfg.PROCESSED_DATA_PATH)
train_ds = train_mod.DeviceDegradationDataset(dataset, dataset["split"]["train"])
val_ds = train_mod.DeviceDegradationDataset(dataset, dataset["split"]["val"])
train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=train_mod.collate_fn)
val_dl = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=train_mod.collate_fn)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = enc_mod.PhysicsEncoder()
        self.decoder = dec_mod.SparsePhysicsDecoder()
        self.ode = ode_mod.PhysicsODE()
        self.alpha_net = ode_mod.DeviceAlphaNet()


model = Model()
ckpt = torch.load(os.path.join(cfg.CHECKPOINT_DIR, "stage2_best.pt"), map_location="cpu")
model_state = model.state_dict()
filtered = {k: v for k, v in ckpt["model_state"].items() if k in model_state}
model.load_state_dict(filtered, strict=False)


def forward_batch(batch):
    z_enc, _ = model.encoder(batch["enc_input"], batch["mask"])
    alpha = model.alpha_net(batch["x0"], batch["T_K"])
    x_hat = model.decoder(z_enc)
    ld = loss_mod.total_physics_loss(
        x_hat, batch["x"], z_enc, batch["T_K"], batch["times_h"], alpha, batch["mask"], model.ode
    )
    ms = loss_mod.multistep_prediction_loss(
        model.decoder, model.ode, z_enc, batch["x"], batch["T_K"], batch["times_h"], alpha, batch["mask"], start_step=0
    )
    return ld, ms, z_enc, alpha

print("INITIAL_SCAN")
for split_name, dl in [("train", train_dl), ("val", val_dl)]:
    for bi, batch in enumerate(dl):
        ld, ms, z, a = forward_batch(batch)
        vals = {k: float(v.detach().cpu()) for k, v in ld.items()}
        vals["ms"] = float(ms.detach().cpu())
        finite = {k: math.isfinite(v) for k, v in vals.items()}
        if not all(finite.values()):
            print(f"nonfinite_forward split={split_name} batch={bi} vals={vals} finite={finite}")
            raise SystemExit(0)
    print(f"all_forward_finite split={split_name}")

print("BACKWARD_SOURCE")
batch = next(iter(train_dl))
ld, ms, z, a = forward_batch(batch)
print("alpha_finite", bool(torch.isfinite(a).all().item()), "alpha_min", float(torch.nan_to_num(a, nan=0.0).min().item()), "alpha_max", float(torch.nan_to_num(a, nan=0.0).max().item()))
print("T_finite", bool(torch.isfinite(batch["T_K"]).all().item()), "x0_finite", bool(torch.isfinite(batch["x0"]).all().item()))
for key in ["ld_total", "ms", "sum"]:
    model.zero_grad(set_to_none=True)
    if key == "ld_total":
        loss = ld["total"]
    elif key == "ms":
        loss = ms
    else:
        loss = ld["total"] + ms
    try:
        loss.backward(retain_graph=True)
    except Exception as exc:
        print(f"{key}: backward_exception={exc}")
        continue
    bad = []
    for n, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            bad.append(n)
    print(f"{key}: finite={len(bad)==0} bad={bad[:8]}")

print("SIMULATE_STAGE3")
params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.ode.parameters()) + list(model.alpha_net.parameters())
opt = torch.optim.Adam(params, lr=cfg.LR_STAGE3)
for epoch in range(1, 4):
    for bi, batch in enumerate(train_dl):
        ld, ms, _, _ = forward_batch(batch)
        loss = ld["total"] + ms
        val = float(loss.detach().cpu())
        if not math.isfinite(val):
            print(f"nan_loss epoch={epoch} batch={bi} ld_total={float(ld['total'])} ms={float(ms)}")
            raise SystemExit(0)
        opt.zero_grad()
        loss.backward()
        grad_bad = []
        for n, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grad_bad.append(n)
        if grad_bad:
            print(f"nan_grad epoch={epoch} batch={bi} bad={grad_bad[:12]}")
            raise SystemExit(0)
        torch.nn.utils.clip_grad_norm_(params, cfg.GRAD_CLIP_NORM)
        opt.step()
        if bi % 5 == 0:
            print(f"epoch={epoch} batch={bi} total={val:.6f} ld_total={float(ld['total']):.6f} ms={float(ms):.6f}")

print("NO_NAN_IN_FIRST_3_EPOCHS")
