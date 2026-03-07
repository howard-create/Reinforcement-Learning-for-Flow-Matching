import fm_dna
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import numpy as np
import oracle
import torch
import torch.nn.functional as F
import wandb
import os
import datetime
from utils import set_seed
from copy import deepcopy
from dataloader_gosai import get_dataloaders_gosai
from omegaconf import OmegaConf


def score(xt, new_model, new_model_y, B, L, gumbel_temp):
    x_out, logits, onehot_tokens, _, _, _ = new_model._sample_finetune(
                    num_sampling_steps = 8,
                    num_samples =  B,
                    sequence_length = L,
                    x = xt,
                    yield_intermediate=True,
                    gumbel_temp = gumbel_temp
                ) 
    preds = new_model_y(onehot_tokens.float().transpose(1, 2))
    reward = preds[:,0]
    return reward

def diff_sample(logits, cfg):
    y_soft = F.softmax(logits / cfg.gumbel_temp, dim=-1)
    idx = y_soft.argmax(dim=-1)
    y_hard = F.one_hot(idx, 4).float()
    onehot_tokens = y_hard.detach() - y_soft.detach() + y_soft

    return onehot_tokens

def save_old_model(model):
    old_model = deepcopy(model)
    old_model.eval()
    for p in old_model.parameters():
        p.requires_grad_(False)
    return old_model



def fine_tune(new_model, new_model_y, new_model_y_eval, old_model, cfg, log_path, save_path, train_loader, val_loader=None, eps=1e-5):
    with open(log_path, 'w') as f:
        f.write(str(cfg) + '\n')
    torch.autograd.set_detect_anomaly(True)

    new_model.config.finetuning.truncate_steps = cfg.truncate_steps
    new_model.config.finetuning.gumbel_softmax_temp = cfg.gumbel_temp
    dt = (1 - eps) / cfg.total_num_steps
    new_model.train()
    torch.set_grad_enabled(True)
    optim = torch.optim.Adam(new_model.parameters(), lr=cfg.learning_rate)
    batch_losses = []
    batch_rewards = []
    before_model = deepcopy(new_model)
    batch = next(iter(train_loader))
    seqs = batch["seqs"].cuda()
    B, L = seqs.shape
    mask_id = 4
    gamma =  0.5
    lam = 1.0
    lastgaelam = 0.0
    delta = 0.2
    #nenv = env.num_envs if hasattr(env, 'num_envs') else 1
    dones = [False for _ in range(B)]
    

    #iteration = 1  #cfg.rollout_times

    for epoch_num in range(cfg.num_epochs):
        rewards = []
        rewards_eval = []
        losses = []
        reward_losses = []
        kl_losses = []
        tot_grad_norm = 0.0
        new_model.train()
        batch_size_per_gpu = new_model.config.loader.eval_batch_size

        optim.zero_grad() 
        for _step in range(cfg.num_accum_steps):
            x = None
            loss = 0.0
            '''
            for i in range(iteration):
                if x is None:
                    stay = torch.full(
                        [B, L],
                        fill_value= 1,
                        dtype=torch.long
                    )
                else:
                    stay = (x == mask_id).long()
                    
                # stay.shape = [2,2]
                stay = stay.unsqueeze(-1).repeat(1, 1, 4)  
                stay = stay.to(device = new_model.device)
            '''

            x_out, logits, onehot_tokens, last_x_list, condt_list, logits_list = new_model._sample_finetune(
                    num_sampling_steps = 8,
                    num_samples =  B,
                    sequence_length = L,    
                    x = x,
                    yield_intermediate=True,
                ) 
            preds = new_model_y(onehot_tokens.float().transpose(1,2))
            reward = preds[:,0]   #terminal reward
            #reward = reward.view(B, 1, 1) 

        
            total_loss = []
            reward_list = []

            mean = reward.mean().detach()
            std  = reward.std(unbiased=False).detach()

            adv = (reward - mean) / (std + 1e-8)

            advantage = adv.view(B, 1, 1) 

            for random_t in range(cfg.total_num_steps):
                #if cfg.truncate_kl and random_t < cfg.total_num_steps - cfg.truncate_steps:
                    #continue
                x_now = last_x_list[random_t]
                condt_now = condt_list[random_t]
                    
                p_x0 = new_model(x_now, condt_now)
                p_x0_old = old_model(x_now, condt_now)
                p_x0 = p_x0[:, :, :-1]
                p_x0_old = p_x0_old[:, :, :-1]
                        
                p_x0 = F.softmax(p_x0, dim = -1) # [B, L, 4]
                p_x0_old = F.softmax(p_x0_old, dim = -1)

                ratio = p_x0 / p_x0_old
                clipped_ratio = torch.clamp(ratio, 1 - delta, 1 + delta) #delta = 0.2
                update_1 = ratio * advantage
                update_2 = clipped_ratio * advantage
                
                update_value = -torch.mean(torch.min(update_1, update_2))


                #log_p_x0 = torch.log(p_x0)
                #log_p_x0_old = torch.log(p_x0_old)
                loss = loss + update_value   #log_p_x0 * lam

            total_loss.append(loss)
            


            '''
            current_alpha = ((epoch_num + 1) / cfg.alpha_schedule_warmup * cfg.alpha) if epoch_num < cfg.alpha_schedule_warmup else cfg.alpha
            kl_loss = torch.stack(total_kl, 1).sum(1).mean()
            reward_loss = -torch.mean(reward)
            loss = #( kl_loss * current_alpha) / (cfg.num_accum_steps) /iteration  # Normalized for gradient accumulation
            loss.backward()
            losses.append(loss.item() * cfg.num_accum_steps*iteration)  # denormalize for logging
            reward_losses.append(reward_loss.item())
            batch_rewards.append(reward.mean().item())
            '''
            
            # Optimizer update after accumulation
            loss =  loss.mean() / cfg.num_accum_steps    #( kl_loss * current_alpha) / (cfg.num_accum_steps) /iteration  # Normalized for gradient accumulation
            loss.backward()
            losses.append(loss.item() * cfg.num_accum_steps)  # denormalize for logging
            if (_step + 1) % cfg.num_accum_steps == 0:
                old_model = save_old_model(new_model)
                norm = torch.nn.utils.clip_grad_norm_(new_model.parameters(), cfg.gradnorm_clip)
                tot_grad_norm += norm
                optim.step()
                optim.zero_grad()

            batch_losses.append(np.sum(losses))

            #batch_rewards.append(torch.mean(reward).cpu().detach().numpy())
            #losses.append(loss.cpu().detach().numpy() * cfg.num_accum_steps)
            #reward_losses.append(reward_loss.cpu().detach().numpy())
            #kl_losses.append(kl_loss.cpu().detach().numpy())

        rewards = np.array(rewards)
        rewards_eval = np.array(rewards_eval)
        losses = np.array(losses)
        reward_losses = np.array(reward_losses)
        kl_losses = np.array(kl_losses)

        print(f"Epoch {epoch_num} Mean reward {np.mean(rewards):.6f} Mean reward eval {np.mean(rewards_eval):.6f} "
                f"Mean grad norm {tot_grad_norm:.6f} Mean loss {np.mean(losses):.6f} "
                f"Mean reward loss {np.mean(reward_losses):.6f} Mean kl loss {np.mean(kl_losses):.6f}")

        if cfg.name != 'debug':
            wandb.log({
                "epoch": epoch_num,
                "mean_reward": np.mean(rewards),
                "mean_reward_eval": np.mean(rewards_eval),
                "mean_grad_norm": tot_grad_norm,
                "mean_loss": np.mean(losses),
                "mean reward loss": np.mean(reward_losses),
                "mean kl loss": np.mean(kl_losses)
            })

        if cfg.validation.enabled and (epoch_num + 1) % cfg.validation.every_n_epochs == 0:
            val_stats = run_validation(new_model, old_model, new_model_y_eval, val_loader or train_loader, cfg, before_model=before_model)
            print(f"[VALID] Epoch {epoch_num+1}: reward {val_stats['val_reward']:.4f}") #, KL {val_stats['val_kl']:.4f}")
            if cfg.validation.log_to_wandb and cfg.name != 'debug':
                wandb.log({
                    "val/reward": val_stats["val_reward"],
                    #"val/kl": val_stats["val_kl"]
                }, step=epoch_num + 1)

        if (epoch_num + 1) % cfg.save_every_n_epochs == 0:
            model_path = os.path.join(save_path, f'model_{epoch_num}.ckpt')
            torch.save(new_model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch_num}")

    if cfg.name != 'debug':
        wandb.finish()

    return batch_losses


@torch.no_grad()
def run_validation(model, old_model, reward_model_eval, dataloader, cfg, before_model):
    model.eval()
    old_model.eval()
    reward_model_eval.eval()

    val_rewards = []
    n = 0
    for i, batch in enumerate(dataloader):
        seqs = batch["seqs"].cuda()
        B, L = seqs.shape
        _, x_out  = model.sample( num_sampling_steps = 8,
        num_samples =  B,
        sequence_length = L,
        yield_intermediate=True,
    )
        tokens = torch.argmax(x_out, dim=-1)
        one_hot = F.one_hot(tokens, num_classes=4).float()
        reward = reward_model_eval(one_hot.transpose(1, 2)).squeeze(-1).mean()
        val_rewards.append(reward.item())
        n += 1

        if cfg.validation._steps == n:
            break

    return {
        "val_reward": sum(val_rewards) / len(val_rewards),
    }

if __name__ == "__main__":
    base_path = "/home/ubuntu" #outputs_gosai/2025.12.27/080231/checkpoints/best.ckpt  
    ckpt_path = os.path.join(base_path, 'mdlm/reward_bp_results_final/alpha0.001_accum4_bsz1_truncate10_temp1.0_clip1.0_test_20260303_072520/model_649.ckpt') 
    log_base_dir = os.path.join(base_path, 'mdlm/reward_bp_results_final')
    GlobalHydra.instance().clear()

    initialize(config_path="configs_gosai", job_name="reward_finetune")
    cfg = compose(config_name="config_gosai.yaml")
    cfg.eval.checkpoint_path = ckpt_path

    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.name == 'debug':
        save_path = os.path.join(log_base_dir, cfg.name)
        os.makedirs(save_path, exist_ok=True)
        log_path = os.path.join(save_path, 'log.txt')
    else:
        run_name = f'alpha{cfg.alpha}_accum{cfg.num_accum_steps}_bsz{cfg.batch_size}_truncate{cfg.truncate_steps}_temp{cfg.gumbel_temp}_clip{cfg.gradnorm_clip}_{cfg.name}_{curr_time}'
        save_path = os.path.join(log_base_dir, run_name)
        os.makedirs(save_path, exist_ok=True)
        wandb.init(
            project='fm_reinforcement',
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=False),
            dir=save_path
        )
        log_path = os.path.join(save_path, 'log.txt')

    set_seed(cfg.seed, use_cuda=True)
    
    #new_model = fm_dna.DiscreteFlowMatchingNet.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)
    #old_model = fm_dna.DiscreteFlowMatchingNet.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)

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

    reward_model = oracle.get_gosai_oracle(mode='train').to(new_model.device)
    reward_model_eval = oracle.get_gosai_oracle(mode='eval').to(new_model.device)
    reward_model.eval()
    reward_model_eval.eval()
    

    train_loader, val_loader, _ = get_dataloaders_gosai(cfg)

    fine_tune(new_model, reward_model, reward_model_eval, old_model, cfg, log_path, save_path, train_loader, val_loader)
