import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from model import ConvNet


def get_timestep_step_sizes(timesteps: torch.Tensor) -> torch.Tensor:
    return -torch.diff(
        timesteps,
        append=torch.zeros([1], device=timesteps.device, dtype=timesteps.dtype),
    )


class DiscreteFlowMatchingNet(pl.LightningModule):
    def __init__(
        self,
        config,
        val_num_sampling_steps: int = 8,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = self.config.fm.vocab_size + 1
        self.hidden_dim = self.config.fm.hidden_dim
        self.num_layers = self.config.fm.num_layers
        self.num_timesteps = self.config.fm.num_timesteps
        self.mask_token_id = 4   
        self.mask_index = 4
        self.val_num_sampling_steps = val_num_sampling_steps
        self.learning_rate = self.config.fm.learning_rate


        self.scheduler = torch.linspace(
            1 / self.num_timesteps, 1, steps=self.num_timesteps, device=self.device, dtype=torch.float32
        )  # Probability path scheduler
        self.dscheduler = torch.linspace(
            1 / self.num_timesteps, 1, steps=self.num_timesteps, device=self.device, dtype=torch.float32
        )
        self.scheduler_type = self.config.fm.scheduler_type
        match self.scheduler_type:
            case "linear":
                pass
            case "square":
                # Put more weight on higher (=more noisy) timesteps.
                # Examples:
                # 0 -> 0 (no noise)
                # 0.5 -> 0.75 (50% noise moved to 75% noise)
                # 1 -> 1 (all noise)
                self.scheduler = 1 - torch.square(1 - self.scheduler)
                self.dscheduler = 2-2*torch.square((self.dscheduler)**(1/2))
            case _:
                raise ValueError(f"Invalid scheduler type: {self.scheduler_type}")

        self.model = ConvNet(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            num_timesteps=self.num_timesteps,
            num_layers=self.num_layers,
        )

        self.save_hyperparameters()

    def on_fit_start(self) -> None:
        self.scheduler = self.scheduler.to(self.device)

        print("Setting learning rate to", self.learning_rate)
        opt = self.optimizers(False)
        assert isinstance(opt, torch.optim.Optimizer)
        opt.param_groups[0]["lr"] = self.learning_rate

    def on_validation_model_eval(self) -> None:
        self.scheduler = self.scheduler.to(self.device)
        return super().on_validation_model_eval()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: B, L, V
        # t: B
        # model: B, L, V
        return self.model(x, t)

    def forward_noising(
        self, x: torch.Tensor, t: torch.Tensor, should_noise: torch.Tensor | None
    ) -> torch.Tensor:
        """Mask x (BL) depending on time step t (BL)."""

        # t is the masking probability. t=0%: dont mask anything, t=100%: mask everything
        mask_prob = self.scheduler[t].expand(-1, x.shape[1]) #sigma
        
        will_mask = torch.bernoulli(mask_prob).to(dtype=torch.bool)

        # Don't mask tokens that should not be noised
        if should_noise is not None:
            will_mask &= should_noise

        noised_x = x.clone()
        noised_x[will_mask] = self.mask_token_id

        return noised_x

    @torch._dynamo.disable
    def log_training_step(self, log_dict):
        self.log_dict(
            {
                **log_dict,
                "train/learning_rate": self.trainer.optimizers[0].param_groups[0]["lr"],
            }
        )

    def _score_entropy(self, log_score, sigma, xt, x0):
        """Computes the SEDD loss.

        Args:
        log_score: float torch.Tensor with shape (batch_size,
            diffusion_model_input_length, vocab_size),
            log score, output of the denoising network.
        xt: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input.
        x0: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input.
        sigma: float torch.Tensor with shape (batch_size, 1).

        Returns:
        loss with shape (batch_size, diffusion_model_input_length)
        """
        # seems that it takes y=x0,xt=M case
        # what is the const term for, seems to be y=M,xt=x0 case and x0 is known so score estimation is precise
        masked_indices = xt == self.mask_index

        expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
        q_ratio = 1 / expsig_minus_1[masked_indices]

        words_that_were_masked = x0[masked_indices]

        neg_term = q_ratio * torch.gather(
        log_score[masked_indices],  
        -1,
        words_that_were_masked[..., None]).squeeze(-1)
        score = log_score[masked_indices].exp()
        if self.mask_index == self.vocab_size - 1:
            pos_term = score[:, :-1].sum(dim=-1)
        else:
            pos_term = score[:, : self.mask_index].sum(
            dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
        const = q_ratio * (q_ratio.log() - 1)

        entropy = torch.zeros(* xt.shape, device=xt.device)
        entropy[masked_indices] += pos_term - neg_term + const
        return entropy
    
    def training_step(self, batch, batch_idx: int):
        # B L
        x = batch["seqs"]
        should_noise: torch.Tensor | None = batch.get("should_noise")
        

        # t: B
        t = torch.randint(0, len(self.scheduler), [x.size(0)], device=x.device)
        self.dscheduler = self.dscheduler.to(self.device)

        sigma = self.scheduler[t]
        dsigma = self.dscheduler[t] 
        # noised_x: B L
        noised_x = self.forward_noising(
            x=x, t=t.unsqueeze(1), should_noise=should_noise
        )

        # Unmasking logits: B L V
        logits = self(noised_x, t)
        logits = logits[:, :, :-1] 

        target = x.clone()
        # Only calculate loss on tokens that were masked
        target[noised_x != self.mask_token_id] = -100
        log_prob = F.log_softmax(logits, dim=-1)

        loss = dsigma[:, None] * self._score_entropy(
            log_prob, sigma[:, None], noised_x, x)

        x = F.one_hot(batch['seqs'], num_classes=4).float()  # [B, L, 4]
        x = x.permute(0, 2, 1)  # → [B, 4, L]

        loss = loss.mean()
        self.log_training_step({"train/loss": loss})

        return loss

    @torch._dynamo.disable
    def log_validation_step(
        self,
        num_samples,
        input_ids,
        generated_ids_seq,
        noised_ids_seq,
        sampling_timesteps,
        losses,
    ):
        def decode_dna(ids):
            vocab = ["A", "C", "G", "T", "[MASK]", "[PAD]"]
            return "".join(vocab[t] for t in ids)

        input_text = [decode_dna(seq) for seq in input_ids]

        generated_texts = [
            [decode_dna(seq) for seq in t_ids]
            for t_ids in generated_ids_seq
        ]

        noised_texts = [
            [decode_dna(seq) for seq in t_ids]
            for t_ids in noised_ids_seq
        ]

        self.log_dict(losses)

        num_samples = min(num_samples, len(input_text))

        if isinstance(self.logger, WandbLogger):
            for i_t, t in enumerate(sampling_timesteps):
                self.logger.log_table(
                    f"validation-texts/{t}",
                    columns=["input_text", "generated_text", "generated_text_inputs"],
                    data=[
                        [
                            input_text[i],
                            generated_texts[i_t][i],
                            noised_texts[i_t][i],
                        ]
                        for i in range(num_samples)
                    ],
                )

    def _get_sampling_timesteps(self, num_sampling_steps):
        return torch.linspace(
            len(self.scheduler) - 1,
            len(self.scheduler) // num_sampling_steps,
            num_sampling_steps,
            device=self.device,
            dtype=torch.long,
        )

    @torch._dynamo.disable
    def validation_step_without_compile(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ):
        x = batch["seqs"]
        should_noise = batch.get("should_noise")

        num_samples = 5  # Number of samples

        sampling_timesteps = self._get_sampling_timesteps(self.val_num_sampling_steps)

        losses = {}

        noised_ids = []
        generated_ids = []

        for t in sampling_timesteps:
            t = t.repeat(x.shape[0])
            assert t.shape == x.shape[:1], t.shape

            noised_x = self.forward_noising(
                x, t.unsqueeze(1), should_noise=should_noise
            )

            target = x.clone()
            target[noised_x != self.mask_token_id] = -100

            logits = self(noised_x, t)
            samples = torch.argmax(logits, dim=-1)

            generated_tokens = noised_x.clone()
            generated_tokens[noised_x == self.mask_token_id] = samples[
                noised_x == self.mask_token_id
            ]

            generated_ids.append(generated_tokens)
            noised_ids.append(noised_x)

            losses[f"validation-losses/loss_{t[0]}"] = F.cross_entropy(
                input=logits.transpose(-1, -2), target=target, reduction="mean"
            )
        losses["validation/loss_mean"] = torch.mean(torch.tensor(list(losses.values())))

        self.log_validation_step(
            num_samples=num_samples,
            input_ids=x,
            generated_ids_seq=generated_ids,
            noised_ids_seq=noised_ids,
            sampling_timesteps=sampling_timesteps,
            losses=losses,
        )

        return losses["validation/loss_mean"]

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss_mean = self.validation_step_without_compile(batch, batch_idx)
        self.log("val/loss", loss_mean)
        return loss_mean
    
    def sample(
        self,
        num_sampling_steps: int,
        num_samples: int | None = None,
        sequence_length: int | None = None,
        x: torch.Tensor | None = None,
        stochasticity: float = 0.0,
        yield_intermediate: bool = False,
        yield_logits: bool = False,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
    ):
        assert (
            num_samples is not None and sequence_length is not None
        ) or x is not None, "Must pass either (num_samples and sequence_length) or x"

        assert not (
            yield_intermediate and yield_logits
        ), "Can't yield both logits and intermediate results"

        # B L
        if x is None:
            # Start fully masked
            x = torch.full(
                [num_samples, sequence_length],
                fill_value=self.mask_token_id,
                dtype=torch.long,
                device=self.device,
            )
            should_noise = None
        else:
            should_noise = x == self.mask_token_id
        self.scheduler = self.scheduler.to(self.device)
        # Create the integer timesteps and step sizes for the given num_sampling_steps
        sampling_timesteps = self._get_sampling_timesteps(num_sampling_steps).to(device=self.device)
        relative_ts = self.scheduler[sampling_timesteps].to(device = self.device)
        relative_dts = get_timestep_step_sizes(relative_ts)

        for t, relative_t, relative_dt in zip(
            sampling_timesteps, relative_ts, relative_dts
        ):
            is_last_step = t == sampling_timesteps[-1]

            t = t.repeat(x.shape[0])
            assert t.shape == x.shape[:1], t.shape

            # B L V
            x_in = x.clone()
            logits = self(x_in, t)

            if cfg_scale != 1.0:
                assert should_noise is not None
                x_uncond = x.clone()
                x_uncond[~should_noise] = self.mask_token_id

                # Classifier-free guidance
                # Run model unconditionally (conditioning fully masked)
                logits_uncond = self(x_uncond, t)

                # Mix the logits according to cfg_scale
                logits = logits_uncond + cfg_scale * (logits - logits_uncond)

            if is_last_step:
                logits = logits[:,:,:-1]

            # B L
            samples = torch.distributions.Categorical(
                logits=logits / temperature
            ).sample()

            # B L
            # Chance to unmask proportional to
            # - step size: higher step size means higher chance
            # - timestep: lower timestep means higher chance (so in the end the chance is 100%)
            unmask_threshold = relative_dt / relative_t

            # With remasking, the unmasking probability is changed
            if stochasticity != 0:
                unmask_threshold *= 1 + stochasticity * (1 - relative_t)

            was_masked = x == self.mask_token_id

            # Unmask
            will_unmask = (
                torch.rand(
                    x.shape[:2],
                    device=unmask_threshold.device,
                    dtype=unmask_threshold.dtype,
                )
                < unmask_threshold
            )
            # Only unmask the tokens that were masked
            will_unmask &= was_masked

            # Remask when stochasticity is non-zero
            if stochasticity != 0 and not is_last_step:
                remask_threshold = relative_dt * stochasticity
                will_remask = (
                    torch.rand(
                        x.shape[:2],
                        device=unmask_threshold.device,
                        dtype=unmask_threshold.dtype,
                    )
                    < remask_threshold
                )
                # Only remask the tokens that were unmasked
                will_remask &= ~was_masked

                # Only remask tokens that aren't constant
                if should_noise is not None:
                    will_remask &= should_noise

                x[will_remask] = self.mask_token_id

            # B L
            x[will_unmask] = samples[will_unmask]

        
        return x, logits

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
    def _sample_finetune(
        self,
        num_sampling_steps: int,
        num_samples: int | None = None,
        sequence_length: int | None = None,
        x: torch.Tensor | None = None,
        stochasticity: float = 0.0,
        yield_intermediate: bool = False,
        yield_logits: bool = False,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        gumbel_temp: float = 1.0
    ):
        assert (
            num_samples is not None and sequence_length is not None
        ) or x is not None, "Must pass either (num_samples and sequence_length) or x"

        assert not (
            yield_intermediate and yield_logits
        ), "Can't yield both logits and intermediate results"

        # B L
        if x is None:
            # Start fully masked
            x = torch.full(
                [num_samples, sequence_length],
                fill_value=self.mask_token_id,
                dtype=torch.long,
                device=self.device,
            )
            should_noise = None
        else:
            should_noise = x == self.mask_token_id
        self.scheduler = self.scheduler.to(self.device)
        # Create the integer timesteps and step sizes for the given num_sampling_steps
        sampling_timesteps = self._get_sampling_timesteps(num_sampling_steps).to(device=self.device)
        relative_ts = self.scheduler[sampling_timesteps].to(device = self.device)
        relative_dts = get_timestep_step_sizes(relative_ts)

        t_list = []
        last_x = []
        logits_list = []

        for t, relative_t, relative_dt in zip(
            sampling_timesteps, relative_ts, relative_dts
        ):
            is_last_step = t == sampling_timesteps[-1]

            t = t.repeat(x.shape[0])
            t_list.append(t)
            assert t.shape == x.shape[:1], t.shape

            # B L V
            x_in = x.clone()
            last_x.append(x_in)
            logits = self(x_in.clone(), t)
            logits_list.append(logits)

            if cfg_scale != 1.0:
                assert should_noise is not None
                x_uncond = x.clone()
                x_uncond[~should_noise] = self.mask_token_id

                # Classifier-free guidance
                # Run model unconditionally (conditioning fully masked)
                logits_uncond = self(x_uncond, t)

                # Mix the logits according to cfg_scale
                logits = logits_uncond + cfg_scale * (logits - logits_uncond)

            if is_last_step:
                logits = logits[:,:,:-1]
                y_soft = F.softmax(logits / gumbel_temp, dim=-1)
                idx = y_soft.argmax(dim=-1)
                y_hard = F.one_hot(idx, 4).float()
                onehot_tokens = y_hard.detach() - y_soft.detach() + y_soft

            # B L
            samples = torch.distributions.Categorical(
                logits=logits / temperature
            ).sample()

            unmask_threshold = relative_dt / relative_t

            # With remasking, the unmasking probability is changed
            if stochasticity != 0:
                unmask_threshold *= 1 + stochasticity * (1 - relative_t)

            was_masked = x == self.mask_token_id

            # Unmask
            will_unmask = (
                torch.rand(
                    x.shape[:2],
                    device=unmask_threshold.device,
                    dtype=unmask_threshold.dtype,
                )
                < unmask_threshold
            )
            # Only unmask the tokens that were masked
            will_unmask &= was_masked

            # Remask when stochasticity is non-zero
            if stochasticity != 0 and not is_last_step:
                remask_threshold = relative_dt * stochasticity
                will_remask = (
                    torch.rand(
                        x.shape[:2],
                        device=unmask_threshold.device,
                        dtype=unmask_threshold.dtype,
                    )
                    < remask_threshold
                )
                # Only remask the tokens that were unmasked
                will_remask &= ~was_masked

                # Only remask tokens that aren't constant
                if should_noise is not None:
                    will_remask &= should_noise

                x[will_remask] = self.mask_token_id

            # B L
            x[will_unmask] = samples[will_unmask]

        
        return x, logits, onehot_tokens, last_x, t_list, logits_list
    
    def prompt_completion(self, seqs, mask_ratio):
        B, L = seqs.size()
        mask_count = int(mask_ratio * L)
        mask_seqs = seqs.clone()

        rand_idx = torch.stack([
            torch.randperm(L, device=seqs.device)[:mask_count] for _ in range(B)
        ])  
        mask_index = 4
        for i in range(B):
            mask_seqs[i, rand_idx[i]] = mask_index
        
        completion_bool = mask_seqs.eq(mask_index)
        prompt_bool = ~completion_bool
        return mask_seqs, prompt_bool, completion_bool