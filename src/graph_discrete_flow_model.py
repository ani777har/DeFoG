import time
import wandb
import os
import json
import random

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions.categorical import Categorical
from hydra.utils import get_original_cwd
from models.transformer_model import GraphTransformer

from metrics.train_metrics import TrainLossDiscrete
from src import utils
from flow_matching.noise_distribution import NoiseDistribution
from flow_matching.time_distorter import TimeDistorter, find_decoding_error_curve
from flow_matching.rate_matrix import RateMatrixDesigner
from flow_matching.utils import p_xt_g_x1
from flow_matching import flow_matching_utils


class GraphDiscreteFlowModel(pl.LightningModule):
    def __init__(
        self,
        cfg,
        dataset_infos,
        train_metrics,
        sampling_metrics,
        visualization_tools,
        extra_features,
        domain_features,
        test_labels=None,
    ):
        super().__init__()

        self.cfg = cfg
        self.name = f"{cfg.dataset.name}_{cfg.general.name}"
        self.model_dtype = torch.float32
        self.conditional = cfg.general.conditional
        self.test_labels = test_labels

        # number of steps used for sampling
        self.sample_T = cfg.sample.sample_steps

        self.input_dims = dataset_infos.input_dims
        self.output_dims = dataset_infos.output_dims
        self.dataset_info = dataset_infos
        self.node_dist = dataset_infos.nodes_dist
        print("max num nodes: ", len(self.node_dist.prob) - 1)
        print("min num nodes: ", torch.where(self.node_dist.prob > 0)[0][0].item())

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.noise_dist = NoiseDistribution(cfg.model.transition, dataset_infos)
        self.limit_dist = self.noise_dist.get_limit_dist()

        # add virtual class when absorbing state refers to a new class
        self.noise_dist.update_input_output_dims(self.input_dims)
        self.noise_dist.update_dataset_infos(self.dataset_info)

        self.train_loss = TrainLossDiscrete(
            self.cfg.model.lambda_train,
        )

        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=self.input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=self.output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        self.save_hyperparameters(
            ignore=[
                "train_metrics",
                "sampling_metrics",
            ],
        )

        # logging
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.val_counter = 0
        self.adapt_counter = 0

        # time distortor for both training and sampling steps
        derived_curve = getattr(cfg.sample, "derived_distortion_curve", None)
        if derived_curve is None:
            # only auto-discover when 'derived' is actually asked for, so a
            # stale curve on disk can never silently change another run
            wants_derived = "derived" in (
                str(cfg.sample.time_distortion),
                str(cfg.train.time_distortion),
            ) or bool(cfg.sample.search)
            if wants_derived:
                derived_curve = find_decoding_error_curve(
                    cfg.dataset.name, root=os.path.join(get_original_cwd(), "..")
                )
        self.time_distorter = TimeDistorter(
            train_distortion=cfg.train.time_distortion,
            sample_distortion=cfg.sample.time_distortion,
            alpha=1,
            beta=1,
            derived_curve_path=derived_curve,
            derived_signal=getattr(
                cfg.sample, "derived_distortion_signal", "soft_combined"
            ),
        )

        # rate matrix designer
        self.rate_matrix_designer = RateMatrixDesigner(
            rdb=self.cfg.sample.rdb,
            rdb_crit=self.cfg.sample.rdb_crit,
            eta=self.cfg.sample.eta,
            omega=self.cfg.sample.omega,
            limit_dist=self.limit_dist,
        )

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return

        if self.conditional:
            if torch.rand(1) < 0.1:
                data.y = torch.ones_like(data.y, device=self.device) * -1

        dense_data, node_mask = utils.to_dense(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=data.y,
            log=i % self.log_every_steps == 0,
        )

        self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=X,
            true_E=E,
            log=i % self.log_every_steps == 0,
        )

        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay,
        )

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print(
            "Size of the input features",
            self.input_dims["X"],
            self.input_dims["E"],
            self.input_dims["y"],
        )
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(
            f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
            f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
            f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
            f" -- {time.time() - self.start_epoch_time:.1f}s "
        )
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(
            f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}"
        )
        if wandb.run:
            wandb.log({"epoch": self.current_epoch}, commit=False)

    def on_validation_epoch_start(self) -> None:
        print("Starting validation...")
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        return

    def on_validation_epoch_end(self) -> None:
        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            print("Starting to sample")
            samples, labels = self.sample(
                is_test=False, save_samples=False, save_visualization=True
            )
            to_log = self.evaluate_samples(
                samples=samples, labels=labels, is_test=False
            )

            # Store results
            filename = os.path.join(
                os.getcwd(),
                f"val_epoch{self.current_epoch}_res_{self.cfg.sample.eta}_{self.cfg.sample.rdb}.txt",
            )
            with open(filename, "w") as file:
                for key, value in to_log.items():
                    file.write(f"{key}: {value}\n")

        self.print("Finished validation.")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.sampling_metrics.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        return

    def on_test_epoch_end(self) -> None:

        if getattr(self.cfg.sample, "measure_decoding_error", False):
            print("Measuring the decoding error curve P_e(t)...")
            self.measure_decoding_error_curve()
        elif self.cfg.sample.search:
            print("Starting sampling optimization...")
            self.search_hyperparameters()
        else:
            print("Starting to sample")
            # Clean generation timing: sync the GPU on both sides so async CUDA
            # work is not misattributed to whatever runs next. NOTE: set
            # general.save_samples=False to keep the (O(n^2) per graph, so
            # dataset-dependent) sample-to-disk writing out of this measurement.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            samples, labels = self.sample(
                is_test=True,
                save_samples=self.cfg.general.save_samples,
                save_visualization=False, # anishok True er
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"[timing] generation: {time.time() - t0:.2f}s for {len(samples)} graphs")
            to_log = self.evaluate_samples(samples=samples, labels=labels, is_test=True)

            # Store results
            filename = os.path.join(
                os.getcwd(),
                f"test_epoch{self.current_epoch}_res_{self.cfg.sample.eta}_{self.cfg.sample.rdb}.txt",
            )
            with open(filename, "w") as file:
                for key, value in to_log.items():
                    file.write(f"{key}: {value}\n")

            self.print("Finished testing.")

    def sample(self, is_test, save_samples, save_visualization):

        # Load generated samples if they exist
        if self.cfg.general.generated_path:
            self.print("Loading generated samples...")
            with open(self.cfg.general.generated_path, "rb") as f:
                samples = pickle.load(f)
            # Set labels to None
            labels = [None] * len(samples)
            return samples, labels

        # Otherwise, generate new samples
        if is_test:
            if self.cfg.general.bootstrapping and self.cfg.general.num_sample_fold != 1:
                raise ValueError(
                    "When bootstrapping is enabled, num_sample_fold must be 1."
                )
            samples_to_generate = (
                self.cfg.general.final_model_samples_to_generate
                * self.cfg.general.num_sample_fold
            )
            samples_left_to_generate = (
                self.cfg.general.final_model_samples_to_generate
                * self.cfg.general.num_sample_fold
            )
            samples_left_to_save = self.cfg.general.final_model_samples_to_save
            chains_left_to_save = self.cfg.general.final_model_chains_to_save

        else:
            samples_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

        samples = []
        labels = []
        graph_id = 0
        while samples_left_to_generate > 0:
            self.print(
                f"Samples left to generate: {samples_left_to_generate}/"
                f"{samples_to_generate}",
                end="",
                flush=True,
            )
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            num_chain_steps = min(self.number_chain_steps, self.sample_T)
            cur_samples, cur_labels = self.sample_batch(
                graph_id,
                to_generate,
                num_nodes=None,
                save_final=to_save,
                keep_chain=chains_save,
                number_chain_steps=num_chain_steps,
                save_visualization=save_visualization,
            )
            samples.extend(cur_samples)
            labels.extend(cur_labels)

            graph_id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        if save_samples:
            self.print("Saving the generated graphs")

            # saving in txt version
            filename = "graphs.txt"
            with open(filename, "w") as f:
                for item in samples:
                    f.write(f"N={item[0].shape[0]}\n")
                    atoms = item[0].tolist()
                    f.write("X: \n")
                    for at in atoms:
                        f.write(f"{at} ")
                    f.write("\n")
                    f.write("E: \n")
                    for bond_list in item[1]:
                        for bond in bond_list:
                            f.write(f"{bond} ")
                        f.write("\n")
                    f.write("\n")

            # saving in pkl version
            with open(f"generated_samples_rank{self.local_rank}.pkl", "wb") as f:
                pickle.dump(samples, f)

            print("Generated graphs saved.")

        return samples, labels

    def evaluate_samples(
        self,
        samples,
        labels,
        is_test,
        save_filename="",
    ):
        print("Computing sampling metrics...")

        to_log = {}
        samples_to_evaluate = self.cfg.general.final_model_samples_to_generate
        if is_test:
            if self.cfg.general.bootstrapping:
                num_bootstrap_fold = 5
                n_total = len(samples)
                holdout = n_total // num_bootstrap_fold
                n_samples_to_evaluate = 40
             
                fold_indices = [
                    np.random.choice(n_total, size=n_samples_to_evaluate, replace=False)
                    for _ in range(num_bootstrap_fold)
                ]
            else:
                fold_indices = [
                    range(i * samples_to_evaluate, (i + 1) * samples_to_evaluate)
                    for i in range(self.cfg.general.num_sample_fold)
                ]

            for i, idx in enumerate(fold_indices):
                cur_samples = [samples[j] for j in idx]
                cur_labels = [labels[j] for j in idx]

                t0 = time.time()
                cur_to_log = self.sampling_metrics.forward(
                    cur_samples,
                    ref_metrics=self.dataset_info.ref_metrics,
                    name=f"self.name_{i}",
                    current_epoch=self.current_epoch,
                    val_counter=-1,
                    test=is_test,
                    local_rank=self.local_rank,
                    labels=cur_labels if self.conditional else None,
                )

                if i == 0:
                    to_log = {i: [cur_to_log[i]] for i in cur_to_log}
                else:
                    to_log = {i: to_log[i] + [cur_to_log[i]] for i in cur_to_log}

                filename = os.path.join(
                    os.getcwd(),
                    f"epoch{self.current_epoch}_res_fold{i}_{save_filename}.txt",
                )
                with open(filename, "w") as file:
                    for key, value in cur_to_log.items():
                        file.write(f"{key}: {value}\n")

                print(f"[timing] eval fold {i}: {time.time() - t0:.2f}s for {len(cur_samples)} graphs")

            to_log = {
                i: (np.array(to_log[i]).mean(), np.array(to_log[i]).std())
                for i in to_log
            }
        else:
            to_log = self.sampling_metrics.forward(
                samples,
                ref_metrics=self.dataset_info.ref_metrics,
                name=self.cfg.general.name,
                current_epoch=self.current_epoch,
                val_counter=-1,
                test=is_test,
                local_rank=self.local_rank,
                labels=labels if self.conditional else None,
            )

        return to_log

    def apply_noise(self, X, E, y, node_mask, t=None):
        """Sample noise and apply it to the data."""

        # Sample a timestep t.
        bs = X.size(0)
        if t is None:
            t_float = self.time_distorter.train_ft(bs, self.device)
        else:
            t_float = t

        # sample random step
        X_1_label = torch.argmax(X, dim=-1)
        E_1_label = torch.argmax(E, dim=-1)
        prob_X_t, prob_E_t = p_xt_g_x1(
            X1=X_1_label, E1=E_1_label, t=t_float, limit_dist=self.limit_dist
        )

        # step 4 - sample noised data
        sampled_t = flow_matching_utils.sample_discrete_features(
            probX=prob_X_t, probE=prob_E_t, node_mask=node_mask
        )
        noise_dims = self.noise_dist.get_noise_dims()
        X_t = F.one_hot(sampled_t.X, num_classes=noise_dims["X"])
        E_t = F.one_hot(sampled_t.E, num_classes=noise_dims["E"])

        # step 5 - create the PlaceHolder
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {
            "t": t_float,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }

        return noisy_data

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(
        self,
        batch_id: int,
        batch_size: int,
        keep_chain: int,
        number_chain_steps: int,
        save_final: int,
        num_nodes=None,
        save_visualization: bool = True,
    ):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(
                batch_size, device=self.device, dtype=torch.int
            )
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()

        # Build the masks
        arange = (
            torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        )
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = flow_matching_utils.sample_discrete_feature_noise(
            limit_dist=self.noise_dist.get_limit_dist(), node_mask=node_mask
        )
        if self.conditional:
            if "qm9" in self.cfg.dataset.name:
                y = self.test_labels
                perm = torch.randperm(y.size(0))
                idx = perm[:100]
                condition = y[idx]
                condition = condition.to(self.device)
                z_T.y = condition.repeat([10, 1])[:batch_size, :]
            elif "tls" in self.cfg.dataset.name:
                z_T.y = torch.zeros(batch_size, 1).to(self.device)
                z_T.y[: batch_size // 2] = 1
            else:
                raise NotImplementedError
        X, E, y = z_T.X, z_T.E, z_T.y

        # Init chain storing variables
        assert (E == torch.transpose(E, 1, 2)).all()
        chain_X_size = torch.Size((number_chain_steps + 1, keep_chain, X.size(1)))
        chain_E_size = torch.Size(
            (number_chain_steps + 1, keep_chain, E.size(1), E.size(2))
        )
        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)
        chain_times = torch.zeros((number_chain_steps + 1, keep_chain))
        chain_time_unit = 1 / number_chain_steps

        # Store initial graph
        if keep_chain > 0:
            sampled_initial = z_T.mask(node_mask, collapse=True)
            chain_X[0] = sampled_initial.X[:keep_chain]
            chain_E[0] = sampled_initial.E[:keep_chain]
            chain_times[0] = torch.zeros((keep_chain))

        for t_int in tqdm(range(0, self.cfg.sample.sample_steps)):
            # this state
            t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
            t_norm = t_array / (self.cfg.sample.sample_steps)
            if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                # to avoid failure mode of absorbing transition, add epsilon
                t_norm = t_norm + 1e-6
            # next state
            s_array = t_array + 1
            s_norm = s_array / (self.cfg.sample.sample_steps)

            # using round for precision
            write_index = int(np.ceil(np.round(s_norm[0].item() / chain_time_unit, 6)))

            # Distort time
            t_norm = self.time_distorter.sample_ft(
                t_norm, self.cfg.sample.time_distortion
            )
            s_norm = self.time_distorter.sample_ft(
                s_norm, self.cfg.sample.time_distortion
            )

            # Sample z_s
            if t_int == 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
            fwd_t0 = time.time()
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                t_norm,
                s_norm,
                X,
                E,
                y,
                node_mask,
            )
            if t_int == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(f"[timing] one forward pass (batch {batch_size}): {(time.time() - fwd_t0) * 1000:.1f} ms")

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]
            chain_times[write_index] = s_norm.flatten()[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0:

            # Repeat last frame 10x to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            chain_times = torch.cat(
                [chain_times, chain_times[-1:].repeat(10, 1)], dim=0
            )
            assert chain_X.size(0) == (number_chain_steps + 1 + 10)

        X, E, y = self.noise_dist.ignore_virtual_classes(X, E, y)
        chain_X, chain_E, _ = self.noise_dist.ignore_virtual_classes(
            chain_X, chain_E, y
        )

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        if self.visualization_tools is not None and save_visualization:
            # Visualize chains
            self.print("Visualizing chains...")
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)  # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(
                    current_path,
                    f"chains/{self.cfg.general.name}/"
                    f"epoch{self.current_epoch}/"
                    f"chains/molecule_{batch_id + i}",
                )
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(
                        result_path,
                        chain_X[:, i, :].numpy(),
                        chain_E[:, i, :].numpy(),
                        chain_times[:, i].numpy(),
                    )
                self.print(
                    "\r{}/{} complete".format(i + 1, num_molecules), end="", flush=True
                )
            self.print("\nVisualizing graphs...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list

    def compute_step_probs(self, R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e):
        step_probs_X = R_t_X * dt  # type: ignore # (B, D, S)
        step_probs_E = R_t_E * dt  # (B, D, S)

        # Calculate the on-diagnoal step probabilities
        # 1) Zero out the diagonal entries
        # assert (E_t.argmax(-1) < 4).all()
        step_probs_X.scatter_(-1, X_t.argmax(-1)[:, :, None], 0.0)
        step_probs_E.scatter_(-1, E_t.argmax(-1)[:, :, :, None], 0.0)

        # 2) Calculate the diagonal entries such that the probability row sums to 1
        step_probs_X.scatter_(
            -1,
            X_t.argmax(-1)[:, :, None],
            (1.0 - step_probs_X.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )
        step_probs_E.scatter_(
            -1,
            E_t.argmax(-1)[:, :, :, None],
            (1.0 - step_probs_E.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )

        # step 2 - merge to the original formulation
        prob_X = step_probs_X.clone()
        prob_E = step_probs_E.clone()

        return prob_X, prob_E

    def sample_p_zs_given_zt(
        self,
        t,
        s,
        X_t,
        E_t,
        y_t,
        node_mask,
        # , condition
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well"""
        bs, n, dx = X_t.shape
        _, _, _, de = E_t.shape
        dt = (s - t)[0]

        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }

        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0
        limit_x = self.limit_dist.X
        limit_e = self.limit_dist.E

        G_1_pred = pred_X, pred_E
        G_t = X_t, E_t

        R_t_X, R_t_E = self.rate_matrix_designer.compute_graph_rate_matrix(
            t,
            node_mask,
            G_t,
            G_1_pred,
        )

        if self.conditional:
            uncond_y = torch.ones_like(y_t, device=self.device) * -1
            noisy_data["y_t"] = uncond_y

            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)

            pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

            R_t_X_uncond, R_t_E_uncond = (
                self.rate_matrix_designer.compute_graph_rate_matrix(
                    t,
                    node_mask,
                    G_t,
                    G_1_pred,
                )
            )

            guidance_weight = self.cfg.general.guidance_weight
            R_t_X = torch.exp(
                torch.log(R_t_X_uncond + 1e-6) * (1 - guidance_weight)
                + torch.log(R_t_X + 1e-6) * guidance_weight
            )
            R_t_E = torch.exp(
                torch.log(R_t_E_uncond + 1e-6) * (1 - guidance_weight)
                + torch.log(R_t_E + 1e-6) * guidance_weight
            )

        prob_X, prob_E = self.compute_step_probs(
            R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e
        )

        if s[0] == 1.0:
            prob_X, prob_E = pred_X, pred_E

        sampled_s = flow_matching_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask
        )

        X_s = F.one_hot(sampled_s.X, num_classes=len(limit_x)).float()
        E_s = F.one_hot(sampled_s.E, num_classes=len(limit_e)).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        if self.conditional:
            y_to_save = y_t
        else:
            y_to_save = torch.zeros([y_t.shape[0], 0], device=self.device)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)

        out_one_hot = out_one_hot.mask(node_mask).type_as(y_t)
        out_discrete = out_discrete.mask(node_mask, collapse=True).type_as(y_t)

        return out_one_hot, out_discrete

    def compute_extra_data(self, noisy_data):
        """At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input."""

        extra_features = self.extra_features(noisy_data)

        # one additional category is added for the absorbing transition
        X, E, y = self.noise_dist.ignore_virtual_classes(
            noisy_data["X_t"], noisy_data["E_t"], noisy_data["y_t"]
        )
        noisy_data_to_mol_feat = noisy_data.copy()
        noisy_data_to_mol_feat["X_t"] = X
        noisy_data_to_mol_feat["E_t"] = E
        noisy_data_to_mol_feat["y_t"] = y
        extra_molecular_features = self.domain_features(noisy_data_to_mol_feat)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data["t"]
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def measure_decoding_error_curve(self):
        """Measure P_e(t), the decoding error rate of this checkpoint.

        Stage 1 of deriving a time schedule from the data instead of picking
        one of the five hand-drawn distortions. Nothing is generated here: real
        graphs are corrupted to a grid of noise levels and reconstructed in one
        forward pass each, so the output is a property of the denoiser and the
        dataset, computed once and cached.

        Runs off the *validation* split by default -- the test split is what
        the sampling metrics are scored against, and a schedule fitted on it
        would be tuned on the evaluation set.
        """
        from analysis import decoding_error

        cfg_s = self.cfg.sample
        split = getattr(cfg_s, "decoding_error_split", "val")
        if split == "val":
            dataloader = self.trainer.datamodule.val_dataloader()
        elif split == "test":
            dataloader = self.trainer.datamodule.test_dataloader()
        elif split == "train":
            dataloader = self.trainer.datamodule.train_dataloader()
        else:
            raise ValueError(f"Unknown decoding_error_split: {split}")

        out_dir = self._search_version_dir(
            "decoding_error", tags=(self.cfg.dataset.name,)
        )

        was_training = self.training
        self.eval()
        try:
            decoding_error.run_and_save(
                self,
                dataloader,
                out_dir=out_dir,
                n_times=getattr(cfg_s, "decoding_error_n_times", 51),
                n_draws=getattr(cfg_s, "decoding_error_n_draws", 8),
                coupled=getattr(cfg_s, "decoding_error_coupled", True),
                max_graphs=getattr(cfg_s, "decoding_error_max_graphs", None),
                seed=getattr(cfg_s, "decoding_error_seed", 0),
                lambda_E=float(self.cfg.model.lambda_train[0]),
            )
        finally:
            if was_training:
                self.train()

        with open(os.path.join(out_dir, "DONE"), "w") as f:
            f.write("decoding error curve measured\n")

    def search_hyperparameters(self):
        """
        Grid search for sampling hypeparameters.
        The num_step_list is tunable based on requirements.
        """

        num_step_list = [50]
        if self.cfg.dataset.name == "qm9":
            num_step_list = [1, 5, 10, 50, 100, 500]
        if self.cfg.dataset.name in ["guacamol", 'moses', 'zinc']:  # accelerate
            num_step_list = [50]

        search_times = {}

        self._search_started_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        self._search_summary_info = []
        self._prior_trial_time_s = 0.0
        self._write_search_summary()

        if self.cfg.sample.search == "all":
            t0 = time.time()
            self.search_distortion(num_step_list)
            search_times["distortion"] = time.time() - t0

            t0 = time.time()
            self.search_stochasticity(num_step_list)
            search_times["stochasticity"] = time.time() - t0

            t0 = time.time()
            self.search_target_guidance(num_step_list)
            search_times["target_guidance"] = time.time() - t0
        elif self.cfg.sample.search == "distortion":
            t0 = time.time()
            self.search_distortion(num_step_list)
            search_times["distortion"] = time.time() - t0

        elif self.cfg.sample.search == "stochasticity":
            t0 = time.time()
            self.search_stochasticity(num_step_list)
            search_times["stochasticity"] = time.time() - t0
        elif self.cfg.sample.search == "target_guidance":
            t0 = time.time()
            self.search_target_guidance(num_step_list)
            search_times["target_guidance"] = time.time() - t0
        elif self.cfg.sample.search == "full_grid":
            t0 = time.time()
            self.search_full_grid(num_step_list)
            search_times["full_grid"] = time.time() - t0
        elif self.cfg.sample.search == "random":
            t0 = time.time()
            self.search_random(num_step_list)
            search_times["random"] = time.time() - t0
        elif self.cfg.sample.search == "bo":
            t0 = time.time()
            self.search_bayesian_optimization(num_step_list)
            search_times["bo"] = time.time() - t0
        elif self.cfg.sample.search == "sobol":
            t0 = time.time()
            self.search_sobol(num_step_list)
            search_times["sobol"] = time.time() - t0
        elif self.cfg.sample.search == "fixed_configs":
            t0 = time.time()
            self.search_fixed_configs()
            search_times["fixed_configs"] = time.time() - t0

        else:
            raise NotImplementedError(
                f"Search type {self.cfg.sample.search} not implemented."
            )
        search_times["total"] = sum(search_times.values())

        self._write_search_summary(search_times)


    def _write_search_summary(self, search_times=None):
        cfg = self.cfg.sample
        with open("search_summary.txt", "w") as f:
            f.write(f"search: {cfg.search}\n")
            f.write(f"status: {'completed' if search_times else 'running'}\n")
            f.write(f"started: {self._search_started_at}\n")
            if cfg.search == "bo":
                f.write(f"search_bo_sampler: {cfg.search_bo_sampler}\n")
                f.write(f"search_n_trials: {cfg.search_n_trials}\n")
                f.write(
                    f"search_bo_n_startup_trials: {cfg.search_bo_n_startup_trials}\n"
                )
                f.write(f"search_seed: {cfg.search_seed}\n")
                f.write(f"search_bo_objective: {cfg.search_bo_objective}\n")
            elif cfg.search == "random":
                f.write(f"search_n_trials: {cfg.search_n_trials}\n")
                f.write(f"eta_range: {list(cfg.search_random_eta_range)}\n")
                f.write(f"omega_range: {list(cfg.search_random_omega_range)}\n")
                f.write(
                    f"search_random_omega_root: {cfg.search_random_omega_root}\n"
                )
                f.write(f"search_seed: {cfg.search_seed}\n")
            elif cfg.search == "sobol":
                f.write("sampler: sobol (optuna QMCSampler, scramble=True)\n")
                f.write(f"search_n_trials: {cfg.search_n_trials}\n")
                f.write(f"search_seed: {cfg.search_seed}\n")
                f.write(f"eta_range: {list(cfg.search_random_eta_range)}\n")
                f.write(f"omega_range: {list(cfg.search_random_omega_range)}\n")
                f.write(
                    f"search_random_omega_root: {cfg.search_random_omega_root}\n"
                )
            elif cfg.search == "fixed_configs":
                f.write("sampler: none (explicit config list, no proposals)\n")

            for line in getattr(self, "_search_summary_info", []):
                f.write(f"{line}\n")

            if search_times:
                f.write("\nTime this run:\n")
                for name, t in search_times.items():
                    f.write(f"{name}: {t:.2f}s ({t / 60:.2f} min)\n")


            # resumed run's time duration info
                prior = getattr(self, "_prior_trial_time_s", 0.0)
                if prior:
                    this_run = search_times.get("total", 0.0)
                    cumulative = prior + this_run
                    f.write("\nIncluding the phase(s) this run continues from:\n")
                    f.write(
                        f"replayed_trial_time: {prior:.2f}s ({prior / 60:.2f} min) "
                        f"[sum of time_s over the replayed trials; their compute "
                        f"only, excluding the earlier job's startup]\n"
                    )
                    f.write(
                        f"cumulative_total: {cumulative:.2f}s "
                        f"({cumulative / 60:.2f} min)\n"
                    )

    def _resume_summary_lines(self, resume_df, resume_path, n_total):
        lines = []
        if resume_path:
            lines.append(f"continued_from: {resume_path}")
            lines.append(
                f"replayed_trials: {len(resume_df)} (ask/tell replayed from CSV, "
                f"no model inference re-run; every proposal verified against the "
                f"recorded one, so the sampler state was exactly reconstructed "
                f"and this continuation is identical to an uninterrupted run)"
            )
        else:
            lines.append("continued_from: (fresh run, not resumed)")
        lines.append(f"executed_trials_this_run: {n_total}")
        return lines

    def _begin_resumable_search(self, resume_df, resume_path, num_step_list, n_trials):
        """Set up the shared resume state for the ask/tell searches (BO, Sobol).

        Returns (n_prior, n_total, trial_idx): completed trials per num_step, the
        number of trials still to run this session, and the next global trial_idx.
        """
        n_prior = {
            ns: (0 if resume_df.empty else int((resume_df["num_step"] == ns).sum()))
            for ns in num_step_list
        }
        n_total = sum(max(0, n_trials - n_prior[ns]) for ns in num_step_list)
        self._prior_trial_time_s = (
            0.0
            if resume_df.empty or "time_s" not in resume_df.columns
            else float(resume_df["time_s"].sum())
        )
        self._search_summary_info = self._resume_summary_lines(
            resume_df, resume_path, n_total
        )
        self._write_search_summary()
        trial_idx = 0 if resume_df.empty else int(resume_df["trial_idx"].max()) + 1
        return n_prior, n_total, trial_idx

    @staticmethod
    def _slugify(value):
        """Make a config value safe for a directory name."""
        text = str(value).strip().lower()
        return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in text)

    def _search_variant_name(self, search_name, tags):
        """
        Directory name for one *variant* of a search.

        Everything that changes which configs get proposed -- dataset, sampler,
        objective, seed -- has to be part of the name. Two runs that share a
        directory share the resume CSV, so a run whose proposals differ would
        otherwise be replayed as if it were the same search.
        """
        parts = [search_name] + [
            self._slugify(t) for t in tags if t is not None and str(t) != ""
        ]
        return "_".join(parts)

    def _search_version_dir(self, search_name, tags=()):
        base_dir = os.path.abspath(
            os.path.join(
                get_original_cwd(),
                "..",
                "outputs",
                self._search_variant_name(search_name, tags),
            )
        )
        os.makedirs(base_dir, exist_ok=True)

        # Claiming a version has to be atomic: two jobs starting together would
        # otherwise both compute the same next_version and share it.
        for _ in range(100):
            existing_versions = sorted(
                int(d.split("_")[1])
                for d in os.listdir(base_dir)
                if d.startswith("version_") and d.split("_")[1].isdigit()
            )

            if existing_versions:
                latest_dir = os.path.join(base_dir, f"version_{existing_versions[-1]}")
                if not os.path.exists(os.path.join(latest_dir, "DONE")):
                    print(f"Resuming search in {latest_dir}")
                    self._record_hydra_run(latest_dir)
                    return latest_dir
                next_version = existing_versions[-1] + 1
            else:
                next_version = 0

            new_dir = os.path.join(base_dir, f"version_{next_version}")
            try:
                os.mkdir(new_dir)
            except FileExistsError:
                # another job claimed this version between the listing and the
                # mkdir -- rescan and try again
                continue
            print(f"Starting search in {new_dir}")
            self._record_hydra_run(new_dir)
            return new_dir

        raise RuntimeError(
            f"Could not claim a version directory under {base_dir} after 100 "
            f"attempts -- too many jobs starting at once?"
        )


    def _record_hydra_run(self, version_dir):
        try:
            from hydra.core.hydra_config import HydraConfig

            run_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
        except Exception:
            # not in a Hydra context (e.g. a unit test) -- fall back to cwd
            run_dir = os.getcwd()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        with open(os.path.join(version_dir, "hydra_runs.txt"), "a") as f:
            f.write(f"{stamp}\t{run_dir}\n")

    def _mark_search_done(self, version_dir):
        open(os.path.join(version_dir, "DONE"), "w").close()

    # ------------------------------------------------------------ probe hooks

    def _probe_set_dir(self, version_dir):
        if self.trajectory_probe is not None:
            self.trajectory_probe.set_output_dir(version_dir)
            print(
                f"[trajectory_probe] logging to "
                f"{self.trajectory_probe.csv_path}"
            )

    def _sample_and_evaluate(self):
        """Generate and evaluate one sampling configuration.

        Generation and evaluation are timed separately: truncating the
        trajectory early only saves the former, so this split is what decides
        whether early stopping is worth anything on a given dataset. On planar
        generation dominates; on SBM the 5-fold bootstrap evaluation does.

        Returns (samples, labels, res, config_time); the two halves are left on
        self._last_sample_time / self._last_eval_time for the caller to log.
        """
        if self.trajectory_probe is not None:
            self.trajectory_probe.begin_trial(
                num_step=self.cfg.sample.sample_steps,
                distortor=self.cfg.sample.time_distortion,
                eta=float(self.cfg.sample.eta),
                omega=float(self.cfg.sample.omega),
            )

        t0 = time.time()
        samples, labels = self.sample(
            is_test=True,
            save_samples=self.cfg.general.save_samples,
            save_visualization=False,
        )
        sample_time = time.time() - t0

        if self.trajectory_probe is not None:
            self.trajectory_probe.end_trial()

        t1 = time.time()
        res = self.evaluate_samples(samples=samples, labels=labels, is_test=True)
        eval_time = time.time() - t1

        # Injected as (mean, std) pairs so the existing
        #   mean_res = {f"{key}_mean": res[key][0] for key in res}
        # in every search picks them up as columns without further changes.
        res["sampling_time_s"] = (sample_time, 0.0)
        res["eval_time_s"] = (eval_time, 0.0)

        self._last_sample_time = sample_time
        self._last_eval_time = eval_time
        print(
            f"  -> generation {sample_time:.2f}s | evaluation {eval_time:.2f}s "
            f"(eval/gen = {eval_time / max(sample_time, 1e-9):.2f})"
        )
        return samples, labels, res, sample_time + eval_time


    def _load_search_checkpoint(self, csv_path, key_cols, dtypes):
        if not os.path.exists(csv_path):
            return pd.DataFrame(), set()
        existing = pd.read_csv(csv_path)
        existing = existing.loc[:, ~existing.columns.str.match(r"^Unnamed")]
        for col, caster in dtypes.items():
            existing[col] = existing[col].apply(caster)
        completed = set(existing[key_cols].apply(tuple, axis=1))
        print(
            f"Resuming from checkpoint {csv_path}: "
            f"{len(completed)} combo(s) already completed."
        )
        return existing, completed

    def _save_search_checkpoint(self, results_df, csv_path):
        tmp_path = f"{csv_path}.tmp"
        results_df.to_csv(tmp_path)
        os.replace(tmp_path, csv_path)


    def search_distortion(self, num_step_list):
        """
        Grid search for sampling distortion.
        """
        results_df = pd.DataFrame()
        distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]
        # distortion_list = ["identity", "polydec"]
        if self.time_distorter.has_derived():
            # the data-derived schedule, benchmarked head to head against the
            # hand-drawn ones under the identical protocol
            distortion_list.append("derived")

        for num_step in num_step_list:
            for distortor in distortion_list:
                self.cfg.sample.sample_steps = num_step
                self.cfg.sample.time_distortion = distortor
                print(
                    f"############# Testing num steps: {num_step}, distortor: {distortor} #############"
                )
                samples, labels, res, config_time = self._sample_and_evaluate()
                print(f"  -> took {config_time:.2f}s")
                mean_res = {f"{key}_mean": res[key][0] for key in res}
                std_res = {f"{key}_std": res[key][1] for key in res}
                mean_res.update(std_res)
                res_df = pd.DataFrame([mean_res])
                res_df["num_step"] = num_step
                res_df["distortor"] = distortor
                res_df["time_s"] = config_time
                results_df = pd.concat([results_df, res_df], ignore_index=True)
                # save at each step as well
                results_df.to_csv(f"search_distortion.csv")

        # set back to default value
        self.cfg.sample.time_distortion = "identity"

        # save the final results
        results_df.reset_index(inplace=True)
        results_df.set_index(["num_step", "distortor"], inplace=True)
        results_df.to_csv(f"search_distortion.csv")

    def search_stochasticity(self, num_step_list):
        """
        Grid search for stochasticity level eta.
        The num_step_list is tunable based on requirements.
        """
        results_df = pd.DataFrame()
        eta_list = [0.0, 5, 10, 25, 50, 100, 200, 300, 500]
        # eta_list = [5, 10]
        for num_step in num_step_list:
            for eta in eta_list:
                self.cfg.sample.sample_steps = num_step
                self.cfg.sample.eta = eta
                self.rate_matrix_designer.eta = eta
                print(
                    f"############# Testing num steps: {num_step}, eta: {eta} #############"
                )
                samples, labels, res, config_time = self._sample_and_evaluate()
                print(f"  -> took {config_time:.2f}s")
                mean_res = {f"{key}_mean": res[key][0] for key in res}
                std_res = {f"{key}_std": res[key][1] for key in res}
                mean_res.update(std_res)
                res_df = pd.DataFrame([mean_res])
                res_df["num_step"] = num_step
                res_df["eta"] = eta
                res_df["time_s"] = config_time
                results_df = pd.concat([results_df, res_df], ignore_index=True)
                # save at each step as well
                results_df.to_csv(f"search_stochasticity.csv")

        # set back to default value
        self.cfg.sample.eta = 0.0
        self.rate_matrix_designer.eta = 0.0

        # save the final results
        results_df.reset_index(inplace=True)
        results_df.set_index(["num_step", "eta"], inplace=True)
        results_df.to_csv(f"search_stochasticity.csv")

    def search_target_guidance(self, num_step_list):
        """
        Grid search for target guidance omega.
        The num_step_list is tunable based on requirements.
        """
        results_df = pd.DataFrame()
        omega_list = [
            0.0,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            1.0,
            2.0,
        ]  # tunable based on requirements
        # omega_list = [0.5, 0.01]  # tunable based on requirements

        for num_step in num_step_list:
            for omega in omega_list:
                self.cfg.sample.sample_steps = num_step
                self.cfg.sample.omega = omega
                self.rate_matrix_designer.omega = omega
                print(
                    f"############# Testing num steps: {num_step}, omega: {omega} #############"
                )
                samples, labels, res, config_time = self._sample_and_evaluate()
                print(f"  -> took {config_time:.2f}s")
                mean_res = {f"{key}_mean": res[key][0] for key in res}
                std_res = {f"{key}_std": res[key][1] for key in res}
                mean_res.update(std_res)
                res_df = pd.DataFrame([mean_res])
                res_df["num_step"] = num_step
                res_df["omega"] = omega
                res_df["time_s"] = config_time
                results_df = pd.concat([results_df, res_df], ignore_index=True)
                # save at each step as well
                results_df.to_csv(f"search_target_guidance.csv")

        # set back to default value
        self.cfg.sample.omega = 0.0
        self.rate_matrix_designer.omega = 0.0

        # save the final results
        results_df.reset_index(inplace=True)
        results_df.set_index(["num_step", "omega"], inplace=True)
        results_df.to_csv(f"search_target_guidance.csv")


    def search_full_grid(self, num_step_list):
        distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]
        eta_list = [0.0, 5, 10, 25, 50, 100]
        omega_list = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

        version_dir = self._search_version_dir(
            "full_grid", tags=(self.cfg.dataset.name,)
        )
        self._probe_set_dir(version_dir)
        checkpoint_path = os.path.join(version_dir, "results.csv")
        key_cols = ["num_step", "distortor", "eta", "omega"]
        dtypes = {
            "num_step": int,
            "distortor": str,
            "eta": float,
            "omega": float,
        }

        results_df, completed = self._load_search_checkpoint(
            checkpoint_path, key_cols, dtypes
        )

        total_runs = (
            len(num_step_list) * len(distortion_list) * len(eta_list) * len(omega_list)
        )

        run_idx = 0
        search_start = time.time()

        for num_step in num_step_list:
            for distortor in distortion_list:
                for eta in eta_list:
                    for omega in omega_list:
                        run_idx += 1
                        if (int(num_step), str(distortor), float(eta), float(omega)) in completed:
                            continue
                        self.cfg.sample.sample_steps = num_step
                        self.cfg.sample.time_distortion = distortor
                        self.cfg.sample.eta = eta
                        self.rate_matrix_designer.eta = eta
                        self.cfg.sample.omega = omega
                        self.rate_matrix_designer.omega = omega
                        print(
                            f"############# [{run_idx}/{total_runs}] Testing num steps: {num_step}, "
                            f"distortor: {distortor}, eta: {eta}, omega: {omega} #############"
                        )
                        samples, labels, res, config_time = self._sample_and_evaluate()
                        elapsed_total = time.time() - search_start
                        avg_run_time = elapsed_total / run_idx
                        eta_remaining = avg_run_time * (total_runs - run_idx)
                        print(
                            f"  -> took {config_time:.2f}s | elapsed: {elapsed_total:.2f}s "
                            f"({elapsed_total / 60:.2f} min) | ETA remaining: "
                            f"{eta_remaining:.2f}s ({eta_remaining / 60:.2f} min)"
                        )
                        mean_res = {f"{key}_mean": res[key][0] for key in res}
                        std_res = {f"{key}_std": res[key][1] for key in res}
                        mean_res.update(std_res)
                        res_df = pd.DataFrame([mean_res])
                        res_df["num_step"] = num_step
                        res_df["distortor"] = distortor
                        res_df["eta"] = eta
                        res_df["omega"] = omega
                        res_df["time_s"] = config_time
                        results_df = pd.concat([results_df, res_df], ignore_index=True)
                        completed.add((int(num_step), str(distortor), float(eta), float(omega)))
                        # save at each step as well
                        self._save_search_checkpoint(results_df, checkpoint_path)

        self.cfg.sample.time_distortion = "identity"
        self.cfg.sample.eta = 0.0
        self.rate_matrix_designer.eta = 0.0
        self.cfg.sample.omega = 0.0
        self.rate_matrix_designer.omega = 0.0

        results_df.reset_index(inplace=True)
        results_df.set_index(["num_step", "distortor", "eta", "omega"], inplace=True)
        self._save_search_checkpoint(results_df, checkpoint_path)
        self._mark_search_done(version_dir)
        print(f"search_full_grid results checkpointed at {checkpoint_path}")


    def _omega_power_transform(self, omega_linear):

        omega_low, omega_high = self.cfg.sample.search_random_omega_range
        span = omega_high - omega_low
        if span == 0:
            return omega_low
        root = self.cfg.sample.search_random_omega_root
        if root == 1.0:
            # Uniform omega: the power map degenerates to the mirror u -> 1-u,
            # so pass the draw through untouched and keep omega == omega_raw.
            return omega_linear
        u = (omega_linear - omega_low) / span
        return omega_low + (1.0 - u ** root) * span

    def search_random(self, num_step_list):

        distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]
        eta_low, eta_high = self.cfg.sample.search_random_eta_range
        omega_low, omega_high = self.cfg.sample.search_random_omega_range
        n_trials = self.cfg.sample.search_n_trials

        version_dir = self._search_version_dir(
            "random",
            tags=(self.cfg.dataset.name, f"seed{self.cfg.sample.search_seed}"),
        )
        self._probe_set_dir(version_dir)
        checkpoint_path = os.path.join(version_dir, "results.csv")
        key_cols = ["num_step", "trial_idx"]
        dtypes = {"num_step": int, "trial_idx": int}
        results_df, completed = self._load_search_checkpoint(
            checkpoint_path, key_cols, dtypes
        )

        rng = random.Random(self.cfg.sample.search_seed)
        n_total = len(num_step_list) * n_trials

        search_start = time.time()
        global_idx = 0
        for num_step in num_step_list:
            for trial_idx in range(n_trials):
                distortor = rng.choice(distortion_list)
                eta = rng.uniform(eta_low, eta_high)
                omega_raw = rng.uniform(omega_low, omega_high)
                omega = self._omega_power_transform(omega_raw)

                if (int(num_step), int(trial_idx)) in completed:
                    global_idx += 1
                    continue

                self.cfg.sample.sample_steps = num_step
                self.cfg.sample.time_distortion = distortor
                self.cfg.sample.eta = eta
                self.rate_matrix_designer.eta = eta
                self.cfg.sample.omega = omega
                self.rate_matrix_designer.omega = omega

                print(
                    f"############# [{global_idx + 1}/{n_total}] Random trial: "
                    f"num_steps: {num_step}, distortor: {distortor}, "
                    f"eta: {eta:.4f}, omega: {omega:.4f} #############"
                )
                samples, labels, res, config_time = self._sample_and_evaluate()
                elapsed_total = time.time() - search_start
                avg_run_time = elapsed_total / (global_idx + 1)
                eta_remaining = avg_run_time * (n_total - global_idx - 1)
                print(
                    f"  -> took {config_time:.2f}s | elapsed: {elapsed_total:.2f}s "
                    f"({elapsed_total / 60:.2f} min) | ETA remaining: "
                    f"{eta_remaining:.2f}s ({eta_remaining / 60:.2f} min)"
                )
                mean_res = {f"{key}_mean": res[key][0] for key in res}
                std_res = {f"{key}_std": res[key][1] for key in res}
                mean_res.update(std_res)
                res_df = pd.DataFrame([mean_res])
                res_df["num_step"] = num_step
                res_df["distortor"] = distortor
                res_df["eta"] = eta
                # omega is the transformed value DeFoG sampled with; omega_raw is
                # the uniform draw before the power transform
                res_df["omega"] = omega
                res_df["omega_raw"] = omega_raw
                res_df["trial_idx"] = trial_idx
                res_df["time_s"] = config_time
                results_df = pd.concat([results_df, res_df], ignore_index=True)
                completed.add((int(num_step), int(trial_idx)))
                self._save_search_checkpoint(results_df, checkpoint_path)
                global_idx += 1

        self.cfg.sample.time_distortion = "identity"
        self.cfg.sample.eta = 0.0
        self.rate_matrix_designer.eta = 0.0
        self.cfg.sample.omega = 0.0
        self.rate_matrix_designer.omega = 0.0
        results_df.reset_index(inplace=True)
        results_df.set_index(
            ["num_step", "distortor", "eta", "omega"], inplace=True
        )
        self._save_search_checkpoint(results_df, checkpoint_path)
        self._mark_search_done(version_dir)
        print(f"search_random results checkpointed at {checkpoint_path}")


    def _load_fixed_configs(self):
        csv_path = self.cfg.sample.search_configs_csv
        csv_path = os.path.abspath(os.path.join(get_original_cwd(), os.path.expanduser(str(csv_path))))

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"sample.search_configs_csv: no such file '{csv_path}'.")

        df = pd.read_csv(csv_path)
        configs = [
            {
                "distortor": str(row["time_distortion"]).strip(),
                "eta": float(row["eta"]),
                "omega": float(row["omega"]),
            }
            for _, row in df.iterrows()
        ]
        return configs, csv_path

    def search_fixed_configs(self):
        """
        Evaluate a fixed list of sampling configs read from a CSV.
        """
        num_step = 1000
        configs, csv_path = self._load_fixed_configs()
        results_df = pd.DataFrame()

        print(
            f"Evaluating {len(configs)} fixed config(s) at num_steps "
            f"{num_step} from {csv_path}"
        )

        for config_idx, config in enumerate(configs):
            distortor = config["distortor"]
            eta = config["eta"]
            omega = config["omega"]

            self.cfg.sample.sample_steps = num_step
            self.cfg.sample.time_distortion = distortor
            self.cfg.sample.eta = eta
            self.rate_matrix_designer.eta = eta
            self.cfg.sample.omega = omega
            self.rate_matrix_designer.omega = omega

            print(
                f"############# Fixed config {config_idx}: "
                f"num_steps: {num_step}, distortor: {distortor}, "
                f"eta: {eta:.4f}, omega: {omega:.4f} #############"
            )

            samples, labels, res, config_time = self._sample_and_evaluate()
            print(f"  -> took {config_time:.2f}s")
            mean_res = {f"{key}_mean": res[key][0] for key in res}
            std_res = {f"{key}_std": res[key][1] for key in res}
            mean_res.update(std_res)

            res_df = pd.DataFrame([mean_res])
            res_df["num_step"] = num_step
            res_df["distortor"] = distortor
            res_df["eta"] = eta
            res_df["omega"] = omega
            res_df["config_idx"] = config_idx
            res_df["time_s"] = config_time
            results_df = pd.concat([results_df, res_df], ignore_index=True)
            # save at each step as well
            results_df.to_csv(f"search_fixed_configs.csv")

        # set back to default values
        self.cfg.sample.time_distortion = "identity"
        self.cfg.sample.eta = 0.0
        self.rate_matrix_designer.eta = 0.0
        self.cfg.sample.omega = 0.0
        self.rate_matrix_designer.omega = 0.0

        # save the final results
        results_df.reset_index(inplace=True)
        results_df.set_index(
            ["num_step", "distortor", "eta", "omega"], inplace=True
        )
        results_df.to_csv(f"search_fixed_configs.csv")

    def _make_bo_sampler(
        self, sampler_name, seed, n_startup_trials, search_space=None, num_obj=1
    ):
        import optuna

        if sampler_name == "gp":
            return optuna.samplers.GPSampler(
                seed=seed, n_startup_trials=n_startup_trials
            )
        elif sampler_name == "tpe":
            return optuna.samplers.TPESampler(
                seed=seed, n_startup_trials=n_startup_trials
            )
        elif sampler_name == "hebo":
            try:
                import optunahub
            except ImportError as e:
                raise ImportError(
                    "search_bo_sampler='hebo' requires the 'optunahub' and "
                    "'hebo' packages (pip install optunahub hebo pymoo; see "
                    "https://hub.optuna.org/samplers/hebo/). "
                ) from e
            hebo_module = optunahub.load_module("samplers/hebo")
            return hebo_module.HEBOSampler(
                search_space=search_space, seed=seed, num_obj=num_obj
            )
        else:
            raise ValueError(f"Unknown search_bo_sampler '{sampler_name}'. ")

    def _save_optuna_visualizations(self, study, num_step, sampler_name, target_names=None):
        try:
            from optuna import visualization as viz
        except Exception as e:
            print(f"  [viz] optuna.visualization unavailable, skipping ({e})")
            return
        if not viz.is_available():
            print("  [viz] plotly not installed; skipping Optuna plots "
                    "(pip install plotly)")
            return
        
        tag = f"{sampler_name}_numstep{num_step}"
        if target_names is None:
            plot_builders = {
                "optimization_history": lambda: viz.plot_optimization_history(study),
                "slice": lambda: viz.plot_slice(study),
            }
        else:
            plot_builders = {
                "pareto_front": lambda: viz.plot_pareto_front(
                    study, target_names=target_names
                ),
            }
            for i, name in enumerate(target_names):
                target = lambda t, i=i: t.values[i]
                plot_builders[f"optimization_history_{name}"] = (
                    lambda target=target, name=name: viz.plot_optimization_history(
                        study, target=target, target_name=name
                    )
                )
                plot_builders[f"slice_{name}"] = (
                    lambda target=target, name=name: viz.plot_slice(
                        study, target=target, target_name=name
                    )
                )
        for name, build in plot_builders.items():
            try:
                fig = build()
            except Exception as e:
                print(f"  [viz] skip {name} for {tag}: {e}")
                continue
            html_path = f"optuna_{tag}_{name}.html"
            try:
                fig.write_html(html_path)
            except Exception as e:
                print(f"  [viz] could not write {html_path}: {e}")
            if wandb.run:
                try:
                    wandb.log(
                        {f"optuna/{name}/{tag}": wandb.Plotly(fig)}, commit=True
                    )
                except Exception as e:
                    print(f"  [viz] could not log {name} to wandb: {e}")
        print(f"  [viz] Optuna plots saved for {tag} (optuna_{tag}_*.html)")


    def _load_search_resume_df(
        self,
        objective_col,
        csv_name="search_bayesian_optimization.csv",
        search_label="bo",
        drop_missing_objective=True,
        auto_checkpoint_path=None,
    ):
        if isinstance(objective_col, str):
            objective_col = [objective_col]

        resume_from = self.cfg.sample.search_bo_resume_from
        if resume_from:
            csv_path = str(resume_from)
            if os.path.isdir(csv_path):
                csv_path = os.path.join(csv_path, csv_name)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"search_bo_resume_from: no {search_label} results CSV found "
                    f"at '{csv_path}'. Resuming a '{search_label}' run needs that "
                    f"run's {csv_name}."
                )
        elif auto_checkpoint_path and os.path.exists(auto_checkpoint_path):
            csv_path = auto_checkpoint_path
            print(
                f"Autoresuming {search_label} search from its checkpoint "
                f"'{csv_path}'."
            )
        else:
            return pd.DataFrame(), None

        df = pd.read_csv(csv_path)
        # Drop what to_csv leaves behind: the unnamed index column, plus the bare
        # 'index' column a *completed* sobol run's reset_index() writes out.
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
        df = df.drop(columns=[c for c in ("index",) if c in df.columns])

        required = {"num_step", "distortor", "eta", "omega", "trial_idx", *objective_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"search_bo_resume_from: '{csv_path}' is missing column(s) "
                f"{sorted(missing)} needed to replay the study. (Was it run with "
                f"a different search_bo_objective?)"
            )
        if drop_missing_objective:
            # A trial missing any objective value can't inform the sampler; drop it.
            df = df.dropna(subset=objective_col).reset_index(drop=True)

        print(
            f"Resuming {search_label} from {csv_path}: {len(df)} completed trial(s) "
            f"will be replayed into the study without re-running model inference."
        )
        return df, csv_path


    def _assert_replay_matches(self, got, recorded, num_step, replay_idx, search_label):
        mismatched = []
        for name, got_val in got.items():
            rec_val = recorded[name]
            if isinstance(got_val, float):
                ok = np.isclose(got_val, float(rec_val), rtol=1e-9, atol=1e-12)
            else:
                ok = str(got_val) == str(rec_val)
            if not ok:
                mismatched.append(name)

        if mismatched:
            rec_str = ", ".join(f"{k}={recorded[k]!r}" for k in got)
            got_str = ", ".join(f"{k}={got[k]!r}" for k in got)
            raise RuntimeError(
                f"{search_label} resume: replayed trial {replay_idx} "
                f"(num_step={num_step}) did not reproduce the recorded proposal "
                f"-- differs in {mismatched}.\n"
                f"  recorded: {rec_str}\n"
                f"  replayed: {got_str}\n"
                f"The sampler's state cannot be reconstructed, so the resume would "
                f"NOT continue the original search. Something the sampler depends on "
                f"differs from the run being resumed -- check that search_bo_sampler, "
                f"search_seed, search_bo_n_startup_trials, search_random_eta_range, "
                f"search_random_omega_range and the optuna version all match it."
            )

    def _replay_bo_trials(self, study, resume_df, num_step, search_space, objective_cols):
        prior = resume_df[resume_df["num_step"] == num_step].sort_values("trial_idx")
        for replay_idx, (_, row) in enumerate(prior.iterrows()):
            with torch.inference_mode(False), torch.enable_grad():
                trial = study.ask(search_space)
            got = {
                "eta": float(trial.params["eta"]),
                "omega": float(trial.params["omega"]),
                "time_distortion": str(trial.params["time_distortion"]),
            }
            recorded = {
                "eta": row["eta"],
                "omega": row["omega"],
                "time_distortion": str(row["distortor"]),
            }
            self._assert_replay_matches(got, recorded, num_step, replay_idx, "BO")
            value = (
                float(row[objective_cols[0]])
                if len(objective_cols) == 1
                else [float(row[c]) for c in objective_cols]
            )
            study.tell(trial, value)
        return len(prior)


    def _sobol_warmup(self, study, search_space):
        import optuna

        warmup_trial = study.ask(search_space)
        study.tell(warmup_trial, state=optuna.trial.TrialState.PRUNED)

    def _replay_sobol_trials(
        self, study, resume_df, num_step, search_space, objective_col, distortion_list
    ):
        import optuna

        prior = resume_df[resume_df["num_step"] == num_step].sort_values("trial_idx")
        n_distortions = len(distortion_list)
        has_u = "distortion_u" in prior.columns
        for replay_idx, (_, row) in enumerate(prior.iterrows()):
            trial = study.ask(search_space)
            got_u = float(trial.params["distortion_u"])
            got = {
                "eta": float(trial.params["eta"]),
                "omega": float(trial.params["omega"]),
            }
            # got["omega"] is the raw sampled value (pre power-transform), so it must
            # be compared against the recorded raw draw, never the transformed omega.
            recorded = {
                "eta": row["eta"],
                "omega": row["omega_raw"],
            }
            if has_u and not pd.isna(row["distortion_u"]):
                got["distortion_u"] = got_u
                recorded["distortion_u"] = row["distortion_u"]
            else:
                got["time_distortion"] = distortion_list[min(int(got_u), n_distortions - 1)]
                recorded["time_distortion"] = str(row["distortor"])
            self._assert_replay_matches(got, recorded, num_step, replay_idx, "sobol")
            value = float(row[objective_col])
            if np.isfinite(value):
                study.tell(trial, value)
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
        return len(prior)

    def search_bayesian_optimization(self, num_step_list):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]
        eta_low, eta_high = self.cfg.sample.search_random_eta_range
        omega_low, omega_high = self.cfg.sample.search_random_omega_range
        n_trials = self.cfg.sample.search_n_trials
        sampler_name = self.cfg.sample.search_bo_sampler
        n_startup_trials = self.cfg.sample.search_bo_n_startup_trials

        objective_specs = {
            "average_ratio": (["average_ratio_mean"], ["minimize"]),
            "vun": (["sampling/frac_unic_non_iso_valid_mean"], ["maximize"]),
            "both": (
                ["average_ratio_mean", "sampling/frac_unic_non_iso_valid_mean"],
                ["minimize", "maximize"],
            ),
        }
        objective_choice = self.cfg.sample.search_bo_objective
        if objective_choice not in objective_specs:
            raise ValueError(
                f"Unknown search_bo_objective '{objective_choice}'. "
                f"Choose from {list(objective_specs)}."
            )
        objective_cols, objective_directions = objective_specs[objective_choice]
        is_multi = len(objective_cols) > 1

        search_space = {
            "eta": optuna.distributions.FloatDistribution(eta_low, eta_high),
            "omega": optuna.distributions.FloatDistribution(omega_low, omega_high),
            "time_distortion": optuna.distributions.CategoricalDistribution(
                distortion_list
            ),
        }

        version_dir = self._search_version_dir(
            "bo",
            tags=(
                self.cfg.dataset.name,
                sampler_name,
                objective_choice,
                f"seed{self.cfg.sample.search_seed}",
            ),
        )
        self._probe_set_dir(version_dir)
        checkpoint_path = os.path.join(
            version_dir, "search_bayesian_optimization.csv"
        )

        resume_df, resume_path = self._load_search_resume_df(
            objective_cols,
            csv_name="search_bayesian_optimization.csv",
            search_label="BO",
            auto_checkpoint_path=checkpoint_path,
        )
        n_prior, n_total, trial_idx = self._begin_resumable_search(
            resume_df, resume_path, num_step_list, n_trials
        )
        results_df = resume_df.copy()
        best_per_num_step = {}
        search_start = time.time()
        executed_idx = 0  # trials actually run this session (drives the ETA)
        for num_step in num_step_list:
            sampler = self._make_bo_sampler(
                sampler_name, self.cfg.sample.search_seed, n_startup_trials,
                search_space=search_space, num_obj=len(objective_cols),
            )
            study = optuna.create_study(
                **(
                    {"direction": objective_directions[0]}
                    if not is_multi
                    else {"directions": objective_directions}
                ),
                sampler=sampler,
            )
            n_done = n_prior[num_step]
            if n_done:
                self._replay_bo_trials(
                    study, resume_df, num_step, search_space, objective_cols
                )
                print(
                    f"Replayed {n_done} completed trial(s) for num_step={num_step} "
                    f"into the {sampler_name} study: every proposal was "
                    f"regenerated and verified against the recorded one, so the "
                    f"sampler state is exactly reconstructed and no model "
                    f"inference was re-run. {max(0, n_trials - n_done)} trial(s) "
                    f"left to run."
                )

            for step_trial_idx in range(n_done, n_trials):
                with torch.inference_mode(False), torch.enable_grad():
                    trial = study.ask(search_space)
                eta = float(trial.params["eta"])
                omega = float(trial.params["omega"])
                distortor = trial.params["time_distortion"]

                self.cfg.sample.sample_steps = num_step
                self.cfg.sample.time_distortion = distortor
                self.cfg.sample.eta = eta
                self.rate_matrix_designer.eta = eta
                self.cfg.sample.omega = omega
                self.rate_matrix_designer.omega = omega

                print(
                    f"############# [{executed_idx + 1}/{n_total}] BO trial "
                    f"({sampler_name}): num_steps: {num_step}, distortor: "
                    f"{distortor}, eta: {eta:.4f}, omega: {omega:.4f} "
                    f"(trial_idx {trial_idx}) #############"
                )
                samples, labels, res, config_time = self._sample_and_evaluate()
                elapsed_total = time.time() - search_start

                avg_run_time = elapsed_total / (executed_idx + 1)
                eta_remaining = avg_run_time * (n_total - executed_idx - 1)
                print(
                    f"  -> took {config_time:.2f}s | elapsed: {elapsed_total:.2f}s "
                    f"({elapsed_total / 60:.2f} min) | ETA remaining: "
                    f"{eta_remaining:.2f}s ({eta_remaining / 60:.2f} min)"
                )
                mean_res = {f"{key}_mean": res[key][0] for key in res}
                std_res = {f"{key}_std": res[key][1] for key in res}
                mean_res.update(std_res)

                value = (
                    float(mean_res[objective_cols[0]])
                    if not is_multi
                    else [float(mean_res[c]) for c in objective_cols]
                )
                study.tell(trial, value)

                res_df = pd.DataFrame([mean_res])
                res_df["num_step"] = num_step
                res_df["distortor"] = distortor
                res_df["eta"] = eta
                res_df["omega"] = omega
                res_df["pair_idx"] = step_trial_idx
                res_df["trial_idx"] = trial_idx
                res_df["time_s"] = config_time
                results_df = pd.concat([results_df, res_df], ignore_index=True)
                self._save_search_checkpoint(results_df, checkpoint_path)

                trial_idx += 1
                executed_idx += 1
            if self.cfg.sample.search_bo_visualize:
                self._save_optuna_visualizations(
                    study, num_step, sampler_name,
                    target_names=(["average_ratio", "vun"] if is_multi else None),
                )

            if study.trials:
                best_per_num_step[num_step] = (
                    [(t.values, t.params) for t in study.best_trials]
                    if is_multi
                    else (study.best_value, study.best_params)
                )


        self.cfg.sample.time_distortion = "identity"
        self.cfg.sample.eta = 0.0
        self.rate_matrix_designer.eta = 0.0
        self.cfg.sample.omega = 0.0
        self.rate_matrix_designer.omega = 0.0

        info = list(self._search_summary_info)
        info.append(f"total_trials_after_run: {len(results_df)}")
        for num_step, entry in best_per_num_step.items():
            if is_multi:
                info.append(
                    f"pareto_front[num_step={num_step}]: {len(entry)} trial(s)"
                )
                for values, params in entry:
                    info.append(
                        f"  avg_ratio={values[0]:.6f}, vun={values[1]:.6f} at "
                        f"eta={params['eta']:.4f}, omega={params['omega']:.4f}, "
                        f"time_distortion={params['time_distortion']}"
                    )
            else:
                best_value, best_params = entry
                info.append(
                    f"best[num_step={num_step}]: {objective_choice}={best_value:.6f} "
                    f"at eta={best_params['eta']:.4f}, omega={best_params['omega']:.4f}, "
                    f"time_distortion={best_params['time_distortion']}"
                )
        self._search_summary_info = info

        results_df.reset_index(inplace=True)
        results_df.set_index(
            ["num_step", "distortor", "eta", "omega"], inplace=True
        )
        self._save_search_checkpoint(results_df, checkpoint_path)
        self._mark_search_done(version_dir)
        print(f"search_bayesian_optimization results checkpointed at {checkpoint_path}")

    def search_sobol(self, num_step_list):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]
        eta_low, eta_high = self.cfg.sample.search_random_eta_range
        omega_low, omega_high = self.cfg.sample.search_random_omega_range
        n_trials = self.cfg.sample.search_n_trials
        objective_col = "average_ratio_mean"

        
        n_distortions = len(distortion_list)
        search_space = {
            "eta": optuna.distributions.FloatDistribution(eta_low, eta_high),
            "omega": optuna.distributions.FloatDistribution(omega_low, omega_high),
            "distortion_u": optuna.distributions.FloatDistribution(0, n_distortions),
        }

        version_dir = self._search_version_dir(
            "sobol",
            tags=(self.cfg.dataset.name, f"seed{self.cfg.sample.search_seed}"),
        )
        self._probe_set_dir(version_dir)
        checkpoint_path = os.path.join(version_dir, "search_sobol.csv")

        resume_df, resume_path = self._load_search_resume_df(
            objective_col,
            csv_name="search_sobol.csv",
            search_label="sobol",
            drop_missing_objective=False,
            auto_checkpoint_path=checkpoint_path,
        )

        n_prior, n_total, trial_idx = self._begin_resumable_search(
            resume_df, resume_path, num_step_list, n_trials
        )
        results_df = resume_df.copy()

        search_start = time.time()
        executed_idx = 0  # trials actually run this session (drives the ETA)
        for num_step in num_step_list:
            
            sampler = optuna.samplers.QMCSampler(
                qmc_type="sobol", scramble=True, seed=self.cfg.sample.search_seed
            )
            study = optuna.create_study(direction="minimize", sampler=sampler)
            self._sobol_warmup(study, search_space)

            n_done = n_prior[num_step]
            if n_done:
                self._replay_sobol_trials(
                    study, resume_df, num_step, search_space, objective_col,
                    distortion_list,
                )
                print(
                    f"Replayed {n_done} completed trial(s) for num_step={num_step} "
                    f"into the sobol study: every point was regenerated and "
                    f"verified against the recorded one, so the sequence resumes "
                    f"at index {n_done} and no model inference was re-run. "
                    f"{max(0, n_trials - n_done)} trial(s) left to run."
                )

            for step_trial_idx in range(n_done, n_trials):
                trial = study.ask(search_space)
                eta = float(trial.params["eta"])
                omega_raw = float(trial.params["omega"])
                omega = self._omega_power_transform(omega_raw)
                
                distortor = distortion_list[
                    min(int(trial.params["distortion_u"]), n_distortions - 1)
                ]
                
                trial.set_user_attr("time_distortion", distortor)

                self.cfg.sample.sample_steps = num_step
                self.cfg.sample.time_distortion = distortor
                self.cfg.sample.eta = eta
                self.rate_matrix_designer.eta = eta
                self.cfg.sample.omega = omega
                self.rate_matrix_designer.omega = omega
                print(
                    f"############# [{executed_idx + 1}/{n_total}] Sobol trial: "
                    f"num_steps: {num_step}, distortor: {distortor}, "
                    f"eta: {eta:.4f}, omega: {omega:.4f} "
                    f"(trial_idx {trial_idx}) #############"
                )
                samples, labels, res, config_time = self._sample_and_evaluate()
                elapsed_total = time.time() - search_start
                
                avg_run_time = elapsed_total / (executed_idx + 1)
                eta_remaining = avg_run_time * (n_total - executed_idx - 1)
                print(
                    f"  -> took {config_time:.2f}s | elapsed: {elapsed_total:.2f}s "
                    f"({elapsed_total / 60:.2f} min) | ETA remaining: "
                    f"{eta_remaining:.2f}s ({eta_remaining / 60:.2f} min)"
                )
                mean_res = {f"{key}_mean": res[key][0] for key in res}
                std_res = {f"{key}_std": res[key][1] for key in res}
                mean_res.update(std_res)

                
                study.tell(trial, float(mean_res[objective_col]))

                if wandb.run:
                    wandb.log(
                        {
                            **mean_res,
                            "num_step": num_step,
                            "distortor": distortor,
                            "eta": eta,
                            "omega": omega,
                            "pair_idx": step_trial_idx,
                            "trial_idx": trial_idx,
                            "time_s": config_time,
                        },
                        commit=True,
                    )
                res_df = pd.DataFrame([mean_res])
                res_df["num_step"] = num_step
                res_df["distortor"] = distortor
                res_df["distortion_u"] = float(trial.params["distortion_u"])
                res_df["eta"] = eta
                res_df["omega"] = omega
                res_df["omega_raw"] = omega_raw
                res_df["pair_idx"] = step_trial_idx
                res_df["trial_idx"] = trial_idx
                res_df["time_s"] = config_time
                results_df = pd.concat([results_df, res_df], ignore_index=True)
                self._save_search_checkpoint(results_df, checkpoint_path)

                trial_idx += 1
                executed_idx += 1

            if self.cfg.sample.search_bo_visualize:
                self._save_optuna_visualizations(study, num_step, "sobol")

        # set back to default values
        self.cfg.sample.time_distortion = "identity"
        self.cfg.sample.eta = 0.0
        self.rate_matrix_designer.eta = 0.0
        self.cfg.sample.omega = 0.0
        self.rate_matrix_designer.omega = 0.0

        # Round out the summary lines published before the search started.
        info = list(self._search_summary_info)
        info.append(f"total_trials_after_run: {len(results_df)}")
        self._search_summary_info = info

        # save the final results
        results_df.reset_index(inplace=True)
        results_df.set_index(
            ["num_step", "distortor", "eta", "omega"], inplace=True
        )
        self._save_search_checkpoint(results_df, checkpoint_path)
        self._mark_search_done(version_dir)
        print(f"search_sobol results checkpointed at {checkpoint_path}")
