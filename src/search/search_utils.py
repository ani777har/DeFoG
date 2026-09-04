"""Support code for the searches in :mod:`search.hyperparameter_search`.

Output directories, resume/checkpoint I/O, the per-trial generate-and-evaluate
step and the Optuna study mechanics. Mixed into ``GraphDiscreteFlowModel``, so
``self`` is the model.
"""

import csv
import os
import time

import numpy as np
import pandas as pd
import torch
import wandb
from hydra.utils import get_original_cwd


class SearchUtilsMixin:
    """Helpers shared by the sampling-hyperparameter searches."""


    def _write_search_summary(self, search_times=None):
        cfg = self.cfg.sample
        with open("search_summary.txt", "w") as f:
            f.write(f"search: {cfg.search}\n")
            f.write(f"status: {'completed' if search_times else 'running'}\n")
            f.write(f"started: {self._search_started_at}\n")
            f.write(f"train_seed: {self.cfg.train.seed}\n")
            if cfg.search == "bo":
                f.write(f"search_bo_sampler: {cfg.search_bo_sampler}\n")
                f.write(f"search_n_trials: {cfg.search_n_trials}\n")
                f.write(
                    f"search_bo_n_startup_trials: {cfg.search_bo_n_startup_trials}\n"
                )
                f.write(f"search_seed: {cfg.search_seed}\n")
                f.write(f"search_bo_objective: {cfg.search_bo_objective}\n")
                f.write(f"eta_range: {list(cfg.search_random_eta_range)}\n")
                f.write(f"omega_range: {list(cfg.search_random_omega_range)}\n")
                f.write(f"distortion_mode: {cfg.search_bo_distortion_mode}\n")
                f.write(
                    f"distortion_a_range: "
                    f"{list(cfg.search_random_distortion_a_range)}\n"
                )
                f.write(
                    f"distortion_b_range: "
                    f"{list(cfg.search_random_distortion_b_range)}\n"
                )
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


    def _axis_search_dir(self, search_name):
        """outputs/<dataset>-<search_name> for the axis searches.

        Flat (no version_N) and reused across runs: all seeds append to the one
        CSV in here (with a 'seed' column), rerunning the axis overwrites it,
        and hydra_runs.txt keeps the pointer to every hydra run directory that
        wrote here.
        """
        out_dir = os.path.abspath(
            os.path.join(
                get_original_cwd(),
                "..",
                "outputs",
                f"{self._slugify(self.cfg.dataset.name)}-{search_name}",
            )
        )
        os.makedirs(out_dir, exist_ok=True)
        print(f"Writing {search_name} results to {out_dir}")
        self._record_hydra_run(out_dir)
        return out_dir

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

    def _sample_and_evaluate(self):
        """Generate and evaluate one sampling configuration.
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


    # the fixed-config CSV schema: 'time' is the time distortion, 'a'/'b' the
    # Kumaraswamy parameters, the rest are descriptive and only carried through
    # to the outputs
    FIXED_CONFIG_HEADER = ["method", "objective", "time", "a", "b", "eta", "omega"]

    @staticmethod
    def _read_fixed_configs_csv(csv_path):
        """Rows are allowed to omit one of the leading descriptive fields (the
        vanilla row carries no 'objective'), so short rows are padded there
        instead of at the end where the numbers live."""
        with open(csv_path, newline="") as f:
            rows = [
                [cell.strip() for cell in row]
                for row in csv.reader(f)
                if any(cell.strip() for cell in row)
            ]
        if not rows:
            raise ValueError(f"sample.search_configs_csv: '{csv_path}' is empty.")

        header, records = rows[0], []
        for line_no, row in enumerate(rows[1:], start=2):
            if len(row) < len(header):
                print(
                    f"  [fixed_configs] line {line_no} of {os.path.basename(csv_path)} has "
                    f"{len(row)} of {len(header)} fields, padding after '{header[0]}'"
                )
                row = row[:1] + [""] * (len(header) - len(row)) + row[1:]
            records.append(dict(zip(header, row[: len(header)])))
        return header, records

    def _load_fixed_configs(self):
        csv_path = self.cfg.sample.search_configs_csv
        csv_path = os.path.abspath(os.path.join(get_original_cwd(), os.path.expanduser(str(csv_path))))

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"sample.search_configs_csv: no such file '{csv_path}'.")

        header, records = self._read_fixed_configs_csv(csv_path)
        missing = [col for col in self.FIXED_CONFIG_HEADER if col not in header]
        if missing:
            raise KeyError(
                f"sample.search_configs_csv: '{csv_path}' is missing column(s) {missing}; "
                f"expected {self.FIXED_CONFIG_HEADER}, found {header}."
            )

        defaults = {"eta": 0.0, "omega": 0.0, "a": 1.0, "b": 1.0}
        configs = []
        for rec in records:
            config = {"distortor": rec["time"]}
            for col, default in defaults.items():
                config[col] = float(rec[col]) if rec[col] != "" else default
            # every CSV column, in its original order, so the descriptive
            # fields (method, objective) reach the outputs
            config["row"] = dict(rec, **{col: config[col] for col in defaults})
            configs.append(config)
        return configs, csv_path, header

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
                "param_importances": lambda: viz.plot_param_importances(study),
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
                plot_builders[f"param_importances_{name}"] = (
                    lambda target=target, name=name: viz.plot_param_importances(
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
        required_cols=("num_step", "distortor", "eta", "omega", "trial_idx"),
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

        required = {*required_cols, *objective_col}
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

    def _bo_distortion_space(self, mode):
        """Optuna distribution(s) for the distortion dimension(s), by mode."""
        import optuna

        if mode == "continuous":
            a_low, a_high = self.cfg.sample.search_random_distortion_a_range
            b_low, b_high = self.cfg.sample.search_random_distortion_b_range
            return {
                "distortion_a": optuna.distributions.FloatDistribution(a_low, a_high),
                "distortion_b": optuna.distributions.FloatDistribution(b_low, b_high),
            }
        elif mode == "categorical":
            distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]
            return {
                "time_distortion": optuna.distributions.CategoricalDistribution(
                    distortion_list
                ),
            }
        else:
            raise ValueError(
                f"Unknown search_bo_distortion_mode '{mode}'. "
                f"Choose 'continuous' or 'categorical'."
            )

    def _bo_apply_distortion(self, trial, mode):
        """Push this trial's distortion param(s) live; return (label, {col: value})."""
        if mode == "continuous":
            a = float(trial.params["distortion_a"])
            b = float(trial.params["distortion_b"])
            self.cfg.sample.time_distortion = "continuous"
            self.cfg.sample.distortion_a = a
            self.time_distorter.distortion_a = a
            self.cfg.sample.distortion_b = b
            self.time_distorter.distortion_b = b
            return f"distortion_a: {a:.4f}, distortion_b: {b:.4f}", {
                "distortion_a": a, "distortion_b": b,
            }
        else:
            d = trial.params["time_distortion"]
            self.cfg.sample.time_distortion = d
            return f"distortor: {d}", {"distortor": d}

    def _bo_reset_distortion(self):
        self.cfg.sample.time_distortion = "identity"
        self.cfg.sample.distortion_a = 1.0
        self.time_distorter.distortion_a = 1.0
        self.cfg.sample.distortion_b = 1.0
        self.time_distorter.distortion_b = 1.0

    def _replay_bo_trials(self, study, resume_df, num_step, search_space, objective_cols, mode):
        prior = resume_df[resume_df["num_step"] == num_step].sort_values("trial_idx")
        for replay_idx, (_, row) in enumerate(prior.iterrows()):
            with torch.inference_mode(False), torch.enable_grad():
                trial = study.ask(search_space)
            got = {"eta": float(trial.params["eta"]), "omega": float(trial.params["omega"])}
            recorded = {"eta": row["eta"], "omega": row["omega"]}
            if mode == "continuous":
                got["distortion_a"] = float(trial.params["distortion_a"])
                got["distortion_b"] = float(trial.params["distortion_b"])
                recorded["distortion_a"] = row["distortion_a"]
                recorded["distortion_b"] = row["distortion_b"]
            else:
                got["time_distortion"] = str(trial.params["time_distortion"])
                recorded["time_distortion"] = str(row["distortor"])
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
