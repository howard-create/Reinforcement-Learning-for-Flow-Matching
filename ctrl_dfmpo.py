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
from dataloader_gosai import get_dataloaders_gosai, index_to_dna
from omegaconf import OmegaConf
import ctrl_regression


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

def load_target_model(cell, device):
        
        oracle_type = "paired"
        base_path = "/mnt/fm_1/datasets"
        model_path = {
            
            'hepg2':f'/mnt/fm_1/datasets/human_enhancers/02_regression_paired/human_regression_paired_hepg2.ckpt',
            'k562': f"/mnt/fm_1/datasets/human_enhancers/02_regression_paired/human_regression_paired_k562.ckpt",
            'sknsh': f"/mnt/fm_1/datasets/human_enhancers/02_regression_paired/human_regression_paired_sknsh.ckpt",
            "JURKAT":f"/human/ckpt/human_{oracle_type}_jurkat.ckpt",
            "K562":f"/human/ckpt/human_{oracle_type}_k562.ckpt",
            "THP1":f"/human/ckpt/human_{oracle_type}_THP1.ckpt",


        }
        model = ctrl_regression.EnformerModel.load_from_checkpoint(
            model_path[cell], map_location='cuda:0'
        ).to(device)
        model.eval()
        return model

def normalize_target(score, cell):
    #fitness_ranges = build_fitness_ranges(cell)
    _, min_fitness, max_fitness = get_fitness_info(cell) #fitness_ranges[cell]
    return (score - min_fitness) / (max_fitness - min_fitness)

#self.fitness_ranges = {cell: get_fitness_info(cell,self.oracle_type) for cell in self.targets.keys()}

def build_fitness_ranges(cell):
    fitness_ranges = {}
    for style in cell:
        fitness_ranges[style] = get_fitness_info(style)
    return fitness_ranges

def get_fitness_info(cell,oracle_type='paired'):
    if oracle_type=='paired' or oracle_type=='dlow':
        if cell == 'hepg2':
            length = 200
            min_fitness = -6.051336
            max_fitness = 10.992575
        elif cell == 'k562':
            length = 200
            min_fitness = -5.857445
            max_fitness = 10.781755
        elif cell == 'sknsh':
            length = 200
            min_fitness = -7.283977
            max_fitness = 12.888308
        elif cell == 'JURKAT':
            length = 250
            min_fitness = -5.574782
            max_fitness =8.555965
        elif cell == 'K562':
            length = 250
            min_fitness = -4.088671
            max_fitness = 10.781755
        elif cell == 'THP1':
            length = 250
            min_fitness = -7.271035
            max_fitness = 6.797082
        else:
            raise NotImplementedError()

    return length, min_fitness, max_fitness

def fine_tune(new_model, reward_model_1, reward_model_2, reward_model_3, new_model_y_eval, old_model, cfg, log_path, save_path, train_loader, val_loader=None, eps=1e-5):
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
    B, _ = seqs.shape
    mask_id = 4
    gamma =  0.5
    lam = 1.0
    lastgaelam = 0.0
    device = new_model.device
    #nenv = env.num_envs if hasattr(env, 'num_envs') else 1
    dones = [False for _ in range(B)]
    if cfg.dna == "enhancer":
        cell = ['hepg2','k562','sknsh']
        reward_model_1 = load_target_model(cell[0], device)
        reward_model_2 = load_target_model(cell[1], device)
        reward_model_3 = load_target_model(cell[2], device)
        L = 200
    else: 
        cell = ['JURKAT', 'K562', 'THP1']
        reward_model_1 = load_target_model(cell[0], device)
        reward_model_2 = load_target_model(cell[1], device)
        reward_model_3 = load_target_model(cell[2], device)
        L = 250


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
            #x_token = index_to_dna(x_out)
            x_token = onehot_tokens

            '''
            reward_list = []
            for i in range(B):
                a = torch.tensor([x_token[i]])
                print(a.shape)
                raw_score_1 = reward_model_1(a).squeeze(0).item()
                norm_score_1 = normalize_target(raw_score_1, cell[0])
                raw_score_2 = reward_model_2(a).squeeze(0).item()
                norm_score_2 = normalize_target(raw_score_2, cell[0])
                raw_score_3 = reward_model_3(a).squeeze(0).item()
                norm_score_3 = normalize_target(raw_score_3, cell[0])
                reward = (norm_score_1 + norm_score_2 + norm_score_3) / 3
                reward_list.append(reward)
            
            reward_list = torch.tensor(reward_list)
            '''
            raw_score_1 = reward_model_1(x_token).squeeze(0)
            norm_score_1 = normalize_target(raw_score_1, cell[0])
            raw_score_2 = reward_model_2(x_token).squeeze(0)
            norm_score_2 = normalize_target(raw_score_2, cell[1])
            raw_score_3 = reward_model_3(x_token).squeeze(0)
            norm_score_3 = normalize_target(raw_score_3, cell[2])
            reward = (norm_score_1 * 0.8 + norm_score_2 * 0.1 + norm_score_3 * 0.1)

            advantage  = reward.view(B, 1, 1)


            
            #preds = new_model_y(onehot_tokens.float().transpose(1,2))
            #reward = preds[:,0]
            #reward = reward.view(B, 1, 1)
            #normal = False
        

            '''        
            preds = new_model_y(onehot_tokens.float().transpose(1, 2))
            reward = preds[:,0]
                    
            sample = x_out
            sample_argmax = 1.0 * F.one_hot(sample, num_classes=4)
            sample_argmax = torch.transpose(sample_argmax, 1, 2)

            preds_argmax = new_model_y(sample_argmax).squeeze(-1)
            reward_argmax = preds_argmax[:, 0]
            rewards.append(reward_argmax.detach().cpu().numpy())

            preds_eval = new_model_y_eval(sample_argmax).squeeze(-1)
            reward_argmax_eval = preds_eval[:, 0]
            rewards_eval.append(reward_argmax_eval.detach().cpu().numpy())
            '''
        
            total_loss = []
            #reward_list = []
            for random_t in range(cfg.total_num_steps):
                #if cfg.truncate_kl and random_t < cfg.total_num_steps - cfg.truncate_steps:
                    #continue

                x_now = last_x_list[random_t]
                condt_now = condt_list[random_t]
                    
                p_x0 = new_model(x_now, condt_now)
                #p_x0_old = old_model(last_x, condt)
                p_x0 = p_x0[:, :, :-1]
                #p_x0_old = p_x0_old[:, :, :-1]
                        
                p_x0 = F.softmax(p_x0, dim = -1) # [B, L, 4]
                #p_x0_old = F.softmax(p_x0_old, dim = -1)
                log_p_x0 = torch.log(p_x0)
                #log_p_x0_old = torch.log(p_x0_old)
                loss = loss + log_p_x0 

            loss = loss * advantage    
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
            val_stats = run_validation(new_model, old_model, reward_model_1, reward_model_2, reward_model_3, cell,  val_loader or train_loader, cfg, before_model=before_model)
            for i in range(len(cell)):
                k = f"val_reward_{cell[i]}"          # 例如 val_reward_HepG2
                r = float(val_stats[k]) 
                print(f"[VALID] Epoch {epoch_num+1}: reward_{cell[i]} {r:.4f}") #, KL {val_stats['val_kl']:.4f}")
                if cfg.validation.log_to_wandb and cfg.name != 'debug':
                    wandb.log({
                        f"val/reward_{cell[i]}": r #f'{val_stats["val_reward{cell[i]}"]}',
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
def run_validation(model, old_model, reward_model_1, reward_model_2, reward_model_3, cell, dataloader, cfg, before_model):
    model.eval()
    old_model.eval()
    #reward_model_eval.eval()

    val_rewards_1 = []
    val_rewards_2 = []
    val_rewards_3 = []
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
        x_token = F.one_hot(tokens, num_classes=4).float()

        raw_score_1 = reward_model_1(x_token).squeeze(-1).mean()
        reward_1 = normalize_target(raw_score_1, cell[0])
        raw_score_2 = reward_model_2(x_token).squeeze(-1).mean()
        reward_2 = normalize_target(raw_score_2, cell[1])
        raw_score_3 = reward_model_3(x_token).squeeze(-1).mean()
        reward_3 = normalize_target(raw_score_3, cell[2])

        '''
        reward_1 = reward_model_eval_1(one_hot).squeeze(-1).mean() #.transpose(1, 2)
        reward_2 = reward_model_eval_2(one_hot).squeeze(-1).mean()
        reward_3 = reward_model_eval_3(one_hot).squeeze(-1).mean()
        '''
        val_rewards_1.append(reward_1.item())
        val_rewards_2.append(reward_2.item())
        val_rewards_3.append(reward_3.item())
        n += 1

        if cfg.validation._steps == n:
            break

    return {
        f"val_reward_{cell[0]}": sum(val_rewards_1) / len(val_rewards_1),
        f"val_reward_{cell[1]}": sum(val_rewards_2) / len(val_rewards_2),
        f"val_reward_{cell[2]}": sum(val_rewards_3) / len(val_rewards_3),
    }

if __name__ == "__main__":
    base_path = "/home/ubuntu"  #outputs_gosai/2025.12.27/080231/checkpoints/best.ckpt
    save_path = "/mnt/fm_1"
    ckpt_path = os.path.join(base_path, 'outputs_gosai/2025.12.27/080231/checkpoints/best.ckpt')  
    log_base_dir = os.path.join(save_path, 'mdlm/reward_bp_results_final')
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
            project='fm_reinforcement_ctrl',
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

    reward_model_1 = None
    reward_model_2 = None
    reward_model_3 = None

    reward_model_eval = None
    #reward_model = oracle.get_gosai_oracle(mode='train').to(new_model.device)
    #reward_model_eval = oracle.get_gosai_oracle(mode='eval').to(new_model.device)
    #reward_model.eval()
    #reward_model_eval.eval()
    

    train_loader, val_loader, _ = get_dataloaders_gosai(cfg)

    fine_tune(new_model, reward_model_1, reward_model_2, reward_model_3, reward_model_eval, old_model, cfg, log_path, save_path, train_loader, val_loader)
