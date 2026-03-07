import os
import datetime
from copy import deepcopy

import torch
import torch.nn.functional as F
import numpy as np
import wandb

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import fm_dna
from utils import set_seed
from dataloader_gosai import get_dataloaders_gosai  
import ctrl_regression

def check_tensor(x, name="tensor"):
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} contains NaN or Inf")


def get_fitness_info(cell, oracle_type="paired"):
    if oracle_type in ["paired", "dlow"]:
        if cell == "hepg2":
            length = 200
            min_fitness = -6.051336
            max_fitness = 10.992575
        elif cell == "k562":
            length = 200
            min_fitness = -5.857445
            max_fitness = 10.781755
        elif cell == "sknsh":
            length = 200
            min_fitness = -7.283977
            max_fitness = 12.888308
        elif cell == "JURKAT":
            length = 250
            min_fitness = -5.574782
            max_fitness = 8.555965
        elif cell == "K562":
            length = 250
            min_fitness = -4.088671
            max_fitness = 10.781755
        elif cell == "THP1":
            length = 250
            min_fitness = -7.271035
            max_fitness = 6.797082
        else:
            raise NotImplementedError(f"Unknown cell: {cell}")
    else:
        raise NotImplementedError(f"Unknown oracle_type: {oracle_type}")

    return length, min_fitness, max_fitness


def normalize_target(score, cell):
    _, min_fitness, max_fitness = get_fitness_info(cell)
    return (score - min_fitness) / (max_fitness - min_fitness)


def load_target_model(cell, device):
    oracle_type = "paired"
    model_path = {
        "hepg2": "/mnt/fm_1/datasets/human_enhancers/02_regression_paired/human_regression_paired_hepg2.ckpt",
        "k562": "/mnt/fm_1/datasets/human_enhancers/02_regression_paired/human_regression_paired_k562.ckpt",
        "sknsh": "/mnt/fm_1/datasets/human_enhancers/02_regression_paired/human_regression_paired_sknsh.ckpt",
        "JURKAT": f"/human/ckpt/human_{oracle_type}_jurkat.ckpt",
        "K562": f"/human/ckpt/human_{oracle_type}_k562.ckpt",
        "THP1": f"/human/ckpt/human_{oracle_type}_THP1.ckpt",
    }
    model = ctrl_regression.EnformerModel.load_from_checkpoint(
        model_path[cell], map_location="cuda:0"
    ).to(device)
    model.eval()
    return model


@torch.no_grad()
def run_validation(model, reward_model_1, reward_model_2, reward_model_3, cell, dataloader, cfg):
    model.eval()

    val_rewards = {c: [] for c in cell}
    n = 0

    for _, batch in enumerate(dataloader):
        seqs = batch["seqs"].cuda()
        B, L = seqs.shape

        _, x_out = model.sample(
            num_sampling_steps=8,
            num_samples=B,
            sequence_length=L,
            yield_intermediate=True,
        )
        tokens = torch.argmax(x_out, dim=-1)
        x_token = F.one_hot(tokens, num_classes=4).float()

        raw_1 = reward_model_1(x_token).squeeze(-1).mean()
        raw_2 = reward_model_2(x_token).squeeze(-1).mean()
        raw_3 = reward_model_3(x_token).squeeze(-1).mean()

        r1 = normalize_target(raw_1, cell[0]).item()
        r2 = normalize_target(raw_2, cell[1]).item()
        r3 = normalize_target(raw_3, cell[2]).item()

        val_rewards[cell[0]].append(r1)
        val_rewards[cell[1]].append(r2)
        val_rewards[cell[2]].append(r3)

        n += 1
        if getattr(cfg.validation, "_steps", 0) and n >= cfg.validation._steps:
            break

    return {f"val_reward_{c}": float(np.mean(val_rewards[c])) for c in cell}


# -----------------------------
# Core: fine_tune with Lagrangian (λ) + thresholds (δ) + target split
# -----------------------------
def fine_tune(
    new_model,
    old_model,
    cfg,
    log_path,
    save_path,
    train_loader,
    val_loader=None,
    eps=1e-5,
):
    # -------- defaults (不要寫回 cfg，避免 OmegaConf struct mode 報錯) --------
    # DictConfig 支援： "key" in cfg 來檢查是否存在
    target_cell = cfg.target_cell if ("target_cell" in cfg) else ("hepg2" if cfg.dna == "enhancer" else "JURKAT")
    delta = list(cfg.delta) if ("delta" in cfg) else [0.5, 0.5]
    lambda_init = list(cfg.lambda_init) if ("lambda_init" in cfg) else [0.1, 0.1]
    lambda_lr = float(cfg.lambda_lr) if ("lambda_lr" in cfg) else 1e-3
    lambda_upper = float(cfg.lambda_upper) if ("lambda_upper" in cfg) else 0.1
    use_hinge = bool(cfg.use_hinge) if ("use_hinge" in cfg) else False

    torch.autograd.set_detect_anomaly(True)

    # diffusion/FM config
    new_model.config.finetuning.truncate_steps = cfg.truncate_steps
    new_model.config.finetuning.gumbel_softmax_temp = cfg.gumbel_temp

    dt = (1 - eps) / cfg.total_num_steps  # (目前沒用到，但先保留)
    new_model.train()
    torch.set_grad_enabled(True)

    device = new_model.device
    optim = torch.optim.Adam(new_model.parameters(), lr=cfg.learning_rate)

    # pick cells / length + load reward models
    if cfg.dna == "enhancer":
        cell = ["hepg2", "k562", "sknsh"]
        L = 200
    else:
        cell = ["JURKAT", "K562", "THP1"]
        L = 250

    reward_model_1 = load_target_model(cell[0], device)
    reward_model_2 = load_target_model(cell[1], device)
    reward_model_3 = load_target_model(cell[2], device)

    # target/off split
    cell2idx = {c: i for i, c in enumerate(cell)}
    if target_cell not in cell2idx:
        raise ValueError(f"target_cell={target_cell} not in {cell}")
    target_idx = cell2idx[target_cell]
    off_indices = [i for i in range(len(cell)) if i != target_idx]  # length 2

    # Lagrangian multipliers (learnable λ1, λ2)
    lambda1 = torch.nn.Parameter(torch.tensor(float(lambda_init[0]), device=device))
    lambda2 = torch.nn.Parameter(torch.tensor(float(lambda_init[1]), device=device))
    lambda_optim = torch.optim.Adam([lambda1, lambda2], lr=lambda_lr)

    # quick batch to get B
    batch = next(iter(train_loader))
    seqs = batch["seqs"].cuda()
    B, _ = seqs.shape

    batch_losses = []
    before_model = deepcopy(new_model)

    for epoch_num in range(cfg.num_epochs):
        losses = []
        tot_grad_norm = 0.0

        new_model.train()
        optim.zero_grad()

        # gradient accumulation loop
        for _step in range(cfg.num_accum_steps):
            x = None
            loss = 0.0

            # --------- sample from diffusion model ---------
            x_out, logits, onehot_tokens, last_x_list, condt_list, logits_list = new_model._sample_finetune(
                num_sampling_steps=8,
                num_samples=B,
                sequence_length=L,
                x=x,
                yield_intermediate=True,
            )

            # tokens for reward model
            x_token = onehot_tokens

            # --------- compute 3 normalized rewards ---------
            raw_1 = reward_model_1(x_token).squeeze(0)
            raw_2 = reward_model_2(x_token).squeeze(0)
            raw_3 = reward_model_3(x_token).squeeze(0)

            norm_1 = normalize_target(raw_1, cell[0])
            norm_2 = normalize_target(raw_2, cell[1])
            norm_3 = normalize_target(raw_3, cell[2])

            scores = [norm_1, norm_2, norm_3]  # aligned with cell list

            target = scores[target_idx]
            off1 = scores[off_indices[0]]
            off2 = scores[off_indices[1]]

            delta1 = float(delta[0])
            delta2 = float(delta[1])

            # constraint violation (off - δ)
            v1 = off1 - delta1
            v2 = off2 - delta2
            if use_hinge:
                v1 = torch.clamp(v1, min=0.0)
                v2 = torch.clamp(v2, min=0.0)

            # --------- dual update: update λ (do NOT backprop into model) ---------
            # increase λ if constraint violated
            lambda_loss = -(
                lambda1 * (off1.mean() - delta1).detach()
                + lambda2 * (off2.mean() - delta2).detach()
            )
            lambda_optim.zero_grad()
            lambda_loss.backward()
            lambda_optim.step()
            with torch.no_grad():
                lambda1.clamp_(min=0.0, max=lambda_upper)
                lambda2.clamp_(min=0.0, max=lambda_upper)

            # --------- Lagrangian reward for policy update ---------
            reward = target - lambda1 * v1 - lambda2 * v2
            advantage = reward.view(B, 1, 1)

            # --------- your original log-prob accumulation ---------
            for random_t in range(cfg.total_num_steps):
                x_now = last_x_list[random_t]
                condt_now = condt_list[random_t]

                p_x0 = new_model(x_now, condt_now)
                p_x0 = p_x0[:, :, :-1]

                check_tensor(p_x0, "p_x0 (pre-softmax)")
                #p_x0 = p_x0.clamp(min=-80, max=80)

                log_p_x0 = F.log_softmax(p_x0, dim=-1)
                #p_x0 = log_p_x0.exp()

                check_tensor(log_p_x0, "log_p_x0")
                loss = loss + log_p_x0

            # maximize reward => minimize negative
            loss = (loss * advantage)

            # --------- backprop into diffusion model ---------
            loss = loss.mean() / cfg.num_accum_steps
            loss.backward()

            losses.append(loss.item() * cfg.num_accum_steps)

            if (_step + 1) % cfg.num_accum_steps == 0:
                norm = torch.nn.utils.clip_grad_norm_(new_model.parameters(), cfg.gradnorm_clip)
                tot_grad_norm += float(norm)
                optim.step()
                optim.zero_grad()

        batch_losses.append(float(np.sum(losses)))

        with torch.no_grad():
            t_mean = float(target.mean().item())
            off1_mean = float(off1.mean().item())
            off2_mean = float(off2.mean().item())
            lam1 = float(lambda1.item())
            lam2 = float(lambda2.item())

        print(
            f"Epoch {epoch_num} | "
            f"target({target_cell})={t_mean:.4f} | "
            f"off1({cell[off_indices[0]]})={off1_mean:.4f} (δ={delta1}) | "
            f"off2({cell[off_indices[1]]})={off2_mean:.4f} (δ={delta2}) | "
            f"λ1={lam1:.4f} λ2={lam2:.4f} | "
            f"grad_norm={tot_grad_norm:.4f} | "
            f"loss={float(np.mean(losses)):.6f}"
        )

        if cfg.name != "debug":
            wandb.log(
                {
                    "epoch": epoch_num,
                    "target_mean": t_mean,
                    f"off_mean_{cell[off_indices[0]]}": off1_mean,
                    f"off_mean_{cell[off_indices[1]]}": off2_mean,
                    "delta1": delta1,
                    "delta2": delta2,
                    "lambda1": lam1,
                    "lambda2": lam2,
                    "mean_grad_norm": tot_grad_norm,
                    "mean_loss": float(np.mean(losses)),
                    "lambda_loss": float(lambda_loss.item()),
                }
            )

        # --------- validation ---------
        if getattr(cfg, "validation", None) and cfg.validation.enabled and (epoch_num + 1) % cfg.validation.every_n_epochs == 0:
            val_stats = run_validation(
                new_model,
                reward_model_1,
                reward_model_2,
                reward_model_3,
                cell,
                val_loader or train_loader,
                cfg,
            )
            for c in cell:
                r = float(val_stats[f"val_reward_{c}"])
                print(f"[VALID] Epoch {epoch_num+1}: reward_{c} {r:.4f}")
                if cfg.validation.log_to_wandb and cfg.name != "debug":
                    wandb.log({f"val/reward_{c}": r}, step=epoch_num + 1)

        # --------- save ---------
        if (epoch_num + 1) % cfg.save_every_n_epochs == 0:
            model_path = os.path.join(save_path, f"model_{epoch_num}.ckpt")
            torch.save(new_model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch_num}: {model_path}")

    if cfg.name != "debug":
        wandb.finish()

    return batch_losses


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    base_path = "/home/ubuntu"
    save_path = "/mnt/fm_1"
    log_base_dir = os.path.join(save_path, "mdlm/reward_bp_results_final")

    # your checkpoint path
    ckpt_path = os.path.join(
        save_path,  #outputs_gosai/2025.12.27/080231/checkpoints/best.ckpt
        "mdlm/reward_bp_results_final/lagrange_delta_target_20260306_072446_alpha0.001_accum4_bsz1_truncate10_temp1.0_clip1.0/model_1599.ckpt", 
    )

    GlobalHydra.instance().clear()
    initialize(config_path="configs_gosai", job_name="reward_finetune")
    cfg = compose(config_name="config_gosai.yaml")
    cfg.eval.checkpoint_path = ckpt_path

    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if cfg.name == "debug":
        run_name = cfg.name
        save_path = os.path.join(log_base_dir, run_name)
        os.makedirs(save_path, exist_ok=True)
        log_path = os.path.join(save_path, "log.txt")
    else:
        run_name = (
            f"lagrange_delta_target_{curr_time}"
            f"_alpha{cfg.alpha}_accum{cfg.num_accum_steps}_bsz{cfg.batch_size}"
            f"_truncate{cfg.truncate_steps}_temp{cfg.gumbel_temp}_clip{cfg.gradnorm_clip}"
        )
        save_path = os.path.join(log_base_dir, run_name)
        os.makedirs(save_path, exist_ok=True)

        wandb.init(
            project="fm_reinforcement_ctrl",
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=False),
            dir=save_path,
        )
        log_path = os.path.join(save_path, "log.txt")

    set_seed(cfg.seed, use_cuda=True)

    if ckpt_path == '/home/ubuntu/outputs_gosai/2025.12.27/080231/checkpoints/best.ckpt':
        new_model = fm_dna.DiscreteFlowMatchingNet.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)
        old_model = fm_dna.DiscreteFlowMatchingNet.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        new_model = fm_dna.DiscreteFlowMatchingNet(config=cfg)
        ckpt = torch.load(cfg.eval.checkpoint_path, map_location="cuda")
        new_model.load_state_dict(ckpt, strict=True)
        new_model = new_model.to(device)

        old_model = fm_dna.DiscreteFlowMatchingNet(config=cfg)
        old_model.load_state_dict(ckpt, strict=True)
        old_model = old_model.to(device)

    train_loader, val_loader, _ = get_dataloaders_gosai(cfg)

    fine_tune(
        new_model=new_model,
        old_model=old_model,
        cfg=cfg,
        log_path=log_path,
        save_path=save_path,
        train_loader=train_loader,
        val_loader=val_loader,
    )
