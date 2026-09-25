"""Sampling-hyperparameter searches over eta, omega and the time distortion.

Entered from ``GraphDiscreteFlowModel.on_test_epoch_end`` when
``cfg.sample.search`` is set. Each ``search_*`` method is a search loop; the
plumbing they run on lives in :mod:`search.search_utils`. Mixed into
``GraphDiscreteFlowModel``, so ``self`` is the model.
"""

import os
import random
import time

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb

from search.search_utils import SearchUtilsMixin

MOLECULAR_DATASETS = ["qm9", "guacamol", "moses", "zinc"]


class HyperparameterSearchMixin(SearchUtilsMixin):
    """The ``search_*`` entry points, dispatched by ``search_hyperparameters``."""

    def search_hyperparameters(self):
        """Run the search named by ``cfg.sample.search``."""
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


    def search_distortion(self, num_step_list):
        """Grid search over the time distortions."""
        distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]
        seed_list = [0, 1, 2]

        out_dir = self._axis_search_dir("distortion")
        self._probe_set_dir(out_dir)
        results_path = os.path.join(out_dir, "search_distortion.csv")
        results_df = pd.DataFrame()

        for seed in seed_list:
            pl.seed_everything(seed)

            for num_step in num_step_list:
                for distortor in distortion_list:
                    self.cfg.sample.sample_steps = num_step
                    self.cfg.sample.time_distortion = distortor
                    print(
                        f"############# Testing num steps: {num_step}, distortor: {distortor}, seed: {seed} #############"
                    )
                    samples, labels, res, config_time = self._sample_and_evaluate()
                    print(f"  -> took {config_time:.2f}s")
                    mean_res = {f"{key}_mean": res[key][0] for key in res}
                    std_res = {f"{key}_std": res[key][1] for key in res}
                    mean_res.update(std_res)
                    res_df = pd.DataFrame([mean_res])
                    res_df["num_step"] = num_step
                    res_df["distortor"] = distortor
                    res_df["seed"] = seed
                    res_df["time_s"] = config_time
                    results_df = pd.concat([results_df, res_df], ignore_index=True)
                    results_df.to_csv(results_path)

        # set back to default value
        self.cfg.sample.time_distortion = "identity"

        # save the final results
        results_df.reset_index(inplace=True)
        results_df.set_index(["num_step", "distortor", "seed"], inplace=True)
        results_df.to_csv(results_path)

    def search_stochasticity(self, num_step_list):
        """Grid search over the stochasticity level eta."""
        eta_list = [0.0, 5, 10, 25, 50, 100, 200, 300, 500]
        seed_list = [0, 1, 2]

        out_dir = self._axis_search_dir("stochasticity")
        self._probe_set_dir(out_dir)
        results_path = os.path.join(out_dir, "search_stochasticity.csv")
        results_df = pd.DataFrame()

        for seed in seed_list:
            pl.seed_everything(seed)

            for num_step in num_step_list:
                for eta in eta_list:
                    self.cfg.sample.sample_steps = num_step
                    self.cfg.sample.eta = eta
                    self.rate_matrix_designer.eta = eta
                    print(
                        f"############# Testing num steps: {num_step}, eta: {eta}, seed: {seed} #############"
                    )
                    samples, labels, res, config_time = self._sample_and_evaluate()
                    print(f"  -> took {config_time:.2f}s")
                    mean_res = {f"{key}_mean": res[key][0] for key in res}
                    std_res = {f"{key}_std": res[key][1] for key in res}
                    mean_res.update(std_res)
                    res_df = pd.DataFrame([mean_res])
                    res_df["num_step"] = num_step
                    res_df["eta"] = eta
                    res_df["seed"] = seed
                    res_df["time_s"] = config_time
                    results_df = pd.concat([results_df, res_df], ignore_index=True)
                    results_df.to_csv(results_path)

        # set back to default value
        self.cfg.sample.eta = 0.0
        self.rate_matrix_designer.eta = 0.0

        # save the final results
        results_df.reset_index(inplace=True)
        results_df.set_index(["num_step", "eta", "seed"], inplace=True)
        results_df.to_csv(results_path)

    def search_target_guidance(self, num_step_list):
        """Grid search over the target guidance omega."""
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
        ]
        seed_list = [0, 1, 2]

        out_dir = self._axis_search_dir("target_guidance")
        self._probe_set_dir(out_dir)
        results_path = os.path.join(out_dir, "search_target_guidance.csv")
        results_df = pd.DataFrame()

        for seed in seed_list:
            pl.seed_everything(seed)

            for num_step in num_step_list:
                for omega in omega_list:
                    self.cfg.sample.sample_steps = num_step
                    self.cfg.sample.omega = omega
                    self.rate_matrix_designer.omega = omega
                    print(
                        f"############# Testing num steps: {num_step}, omega: {omega}, seed: {seed} #############"
                    )
                    samples, labels, res, config_time = self._sample_and_evaluate()
                    print(f"  -> took {config_time:.2f}s")
                    mean_res = {f"{key}_mean": res[key][0] for key in res}
                    std_res = {f"{key}_std": res[key][1] for key in res}
                    mean_res.update(std_res)
                    res_df = pd.DataFrame([mean_res])
                    res_df["num_step"] = num_step
                    res_df["omega"] = omega
                    res_df["seed"] = seed
                    res_df["time_s"] = config_time
                    results_df = pd.concat([results_df, res_df], ignore_index=True)
                    results_df.to_csv(results_path)

        # set back to default value
        self.cfg.sample.omega = 0.0
        self.rate_matrix_designer.omega = 0.0

        # save the final results
        results_df.reset_index(inplace=True)
        results_df.set_index(["num_step", "omega", "seed"], inplace=True)
        results_df.to_csv(results_path)


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

    def search_fixed_configs(self):
        """Evaluate a fixed list of sampling configs read from a CSV."""
        num_step_list = [5, 10, 25, 50, 100, 250, 500, 1000]
        sample_size_list = [self.cfg.general.final_model_samples_to_generate]
        seed_list = [0, 1, 2]
        configs, csv_path, csv_header = self._load_fixed_configs()
        results_df = pd.DataFrame()

        # one row per config, holding every column of the source CSV, to be
        # joined back onto the per-config/n_samples aggregates
        configs_df = pd.DataFrame(
            [dict(config["row"], config_idx=idx) for idx, config in enumerate(configs)]
        )
        # [trial], method, objective, time, a, b, eta, omega, config_idx,
        # num_step, n_samples, stats
        stats_cols = csv_header + ["config_idx", "num_step", "n_samples"]
        label_cols = self._fixed_config_label_cols(csv_header)

        # writes in the hydra run directory
        self._probe_set_dir(os.getcwd())

        print(
            f"Evaluating {len(configs)} fixed config(s) at num_steps {num_step_list}, "
            f"sample sizes {sample_size_list}, seeds {seed_list} from {csv_path}"
        )

        for config_idx, config in enumerate(configs):
            distortor = config["distortor"]
            eta = config["eta"]
            omega = config["omega"]

            for num_step in num_step_list:
                for n_samples in sample_size_list:
                    for seed in seed_list:
                        pl.seed_everything(seed)
                        self.cfg.sample.sample_steps = num_step
                        self.cfg.general.final_model_samples_to_generate = n_samples
                        self.cfg.sample.time_distortion = distortor
                        self.cfg.sample.eta = eta
                        self.rate_matrix_designer.eta = eta
                        self.cfg.sample.omega = omega
                        self.rate_matrix_designer.omega = omega
                        self.cfg.sample.distortion_a = config["a"]
                        self.time_distorter.distortion_a = config["a"]
                        self.cfg.sample.distortion_b = config["b"]
                        self.time_distorter.distortion_b = config["b"]

                        label = "/".join(
                            str(config["row"][col]) for col in label_cols
                        )
                        print(
                            f"############# Fixed config {config_idx} "
                            f"({label}): "
                            f"n_samples: {n_samples}, num_steps: {num_step}, "
                            f"distortor: {distortor}, "
                            f"eta: {eta:.4f}, omega: {omega:.4f}, "
                            f"a: {config['a']:.4f}, b: {config['b']:.4f}, "
                            f"seed: {seed} #############"
                        )

                        samples, labels, res, config_time = self._sample_and_evaluate()
                        print(f"  -> took {config_time:.2f}s")
                        mean_res = {f"{key}_mean": res[key][0] for key in res}
                        std_res = {f"{key}_std": res[key][1] for key in res}
                        mean_res.update(std_res)

                        res_df = pd.DataFrame([mean_res])
                        res_df["n_samples"] = n_samples
                        res_df["num_step"] = num_step
                        res_df["distortor"] = distortor
                        res_df["eta"] = eta
                        res_df["omega"] = omega
                        res_df["distortion_a"] = config["a"]
                        res_df["distortion_b"] = config["b"]
                        for col in label_cols:
                            res_df[col] = config["row"][col]
                        res_df["config_idx"] = config_idx
                        res_df["seed"] = seed
                        res_df["time_s"] = config_time
                        results_df = pd.concat([results_df, res_df], ignore_index=True)
                        results_df.to_csv(f"search_fixed_configs.csv")
                        # mean/sd over the seeds, per config and sample size
                        seed_aggs = dict(
                            vun_mean=("sampling/frac_unic_non_iso_valid_mean", "mean"),
                            vun_sd=("sampling/frac_unic_non_iso_valid_mean", "std"),
                        )
                        # no average_ratio on molecular datasets, vun only
                        if self.cfg.dataset.name not in MOLECULAR_DATASETS:
                            seed_aggs.update(
                                avg_ratio_mean=("average_ratio_mean", "mean"),
                                avg_ratio_sd=("average_ratio_mean", "std"),
                            )
                        seed_stats = (
                            results_df.groupby(["config_idx", "num_step", "n_samples"])
                            .agg(**seed_aggs)
                            .reset_index()
                        )
                        # prepend the source CSV columns describing each config
                        seed_stats = configs_df.merge(seed_stats, on="config_idx")
                        seed_stats = seed_stats[stats_cols + list(seed_aggs)]
                        seed_stats.to_csv(
                            "search_fixed_configs_seed_stats.csv", index=False
                        )

        # set back to default values
        self.cfg.sample.time_distortion = "identity"
        self.cfg.sample.eta = 0.0
        self.rate_matrix_designer.eta = 0.0
        self.cfg.sample.omega = 0.0
        self.rate_matrix_designer.omega = 0.0

        # save the final results
        results_df.reset_index(inplace=True)
        results_df.set_index(
            ["num_step", "n_samples", "distortor", "eta", "omega", "seed"], inplace=True
        )
        results_df.to_csv(f"search_fixed_configs.csv")

    def search_bayesian_optimization(self, num_step_list):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        distortion_mode = getattr(self.cfg.sample, "search_bo_distortion_mode", "continuous")
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
        # Molecular datasets produce no average_ratio: fcd is their only ratio
        # metric and it is disabled in MolecularSamplingMetrics, so vun is the
        # only objective with a column to read.
        if self.cfg.dataset.name in MOLECULAR_DATASETS and objective_choice != "vun":
            print(
                f"search_bo_objective '{objective_choice}' needs average_ratio, which "
                f"{self.cfg.dataset.name} does not produce. Using 'vun' instead."
            )
            objective_choice = "vun"
        objective_cols, objective_directions = objective_specs[objective_choice]
        is_multi = len(objective_cols) > 1

        search_space = {
            "eta": optuna.distributions.FloatDistribution(eta_low, eta_high),
            "omega": optuna.distributions.FloatDistribution(omega_low, omega_high),
            **self._bo_distortion_space(distortion_mode),
        }

        version_dir = self._search_version_dir(
            "bo",
            tags=(
                self.cfg.dataset.name,
                sampler_name,
                objective_choice,
                distortion_mode,
                f"seed{self.cfg.sample.search_seed}",
            ),
        )
        self._probe_set_dir(version_dir)
        checkpoint_path = os.path.join(
            version_dir, "search_bayesian_optimization.csv"
        )

        distortion_cols = (
            ("distortion_a", "distortion_b") if distortion_mode == "continuous"
            else ("distortor",)
        )
        resume_df, resume_path = self._load_search_resume_df(
            objective_cols,
            csv_name="search_bayesian_optimization.csv",
            search_label="BO",
            auto_checkpoint_path=checkpoint_path,
            required_cols=("num_step", *distortion_cols, "eta", "omega", "trial_idx"),
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
                    study, resume_df, num_step, search_space, objective_cols, distortion_mode
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
                distortion_label, distortion_cols_val = self._bo_apply_distortion(
                    trial, distortion_mode
                )

                self.cfg.sample.sample_steps = num_step
                self.cfg.sample.eta = eta
                self.rate_matrix_designer.eta = eta
                self.cfg.sample.omega = omega
                self.rate_matrix_designer.omega = omega

                print(
                    f"############# [{executed_idx + 1}/{n_total}] BO trial "
                    f"({sampler_name}): num_steps: {num_step}, {distortion_label}, "
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

                value = (
                    float(mean_res[objective_cols[0]])
                    if not is_multi
                    else [float(mean_res[c]) for c in objective_cols]
                )
                study.tell(trial, value)

                res_df = pd.DataFrame([mean_res])
                res_df["num_step"] = num_step
                for col, val in distortion_cols_val.items():
                    res_df[col] = val
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


        self._bo_reset_distortion()
        self.cfg.sample.eta = 0.0
        self.rate_matrix_designer.eta = 0.0
        self.cfg.sample.omega = 0.0
        self.rate_matrix_designer.omega = 0.0

        def _distortion_str(params):
            if distortion_mode == "continuous":
                return (
                    f"distortion_a={params['distortion_a']:.4f}, "
                    f"distortion_b={params['distortion_b']:.4f}"
                )
            return f"time_distortion={params['time_distortion']}"

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
                        f"{_distortion_str(params)}"
                    )
            else:
                best_value, best_params = entry
                info.append(
                    f"best[num_step={num_step}]: {objective_choice}={best_value:.6f} "
                    f"at eta={best_params['eta']:.4f}, omega={best_params['omega']:.4f}, "
                    f"{_distortion_str(best_params)}"
                )
        self._search_summary_info = info

        results_df.reset_index(inplace=True)
        results_df.set_index(
            ["num_step", *distortion_cols, "eta", "omega"], inplace=True
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
