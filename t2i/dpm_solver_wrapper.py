import torch
from tqdm import tqdm
import os

from diffusion.model.dpm_solver import DPM_Solver


class DPM_Solver_Wrapper(DPM_Solver):

    def sample(
        self,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=2,
        skip_type="time_uniform",
        method="multistep",
        lower_order_final=True,
        denoise_to_zero=False,
        solver_type="dpmsolver",
        atol=0.0078,
        rtol=0.05,
        return_intermediate=False,
        flow_shift=1.0,
    ):
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert (
            t_0 > 0 and t_T > 0
        ), "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in [
                "multistep",
                "singlestep",
                "singlestep_fixed",
            ], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in [
                "multistep",
                "singlestep",
                "singlestep_fixed",
            ], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []
        with torch.no_grad():
            if method == "adaptive":
                x = self.dpm_solver_adaptive(
                    x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol, solver_type=solver_type
                )
            elif method == "multistep":
                assert steps >= order
                timesteps = self.get_time_steps(
                    skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device, shift=flow_shift
                )
                assert timesteps.shape[0] - 1 == steps
                # Init the initial values.
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)
                self.update_progress(step + 1, len(timesteps))
                # Init the first `order` values by lower order multistep DPM-Solver.
                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_dpm_solver_update(
                        x, model_prev_list, t_prev_list, t, step, solver_type=solver_type
                    )
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))
                    # update progress bar
                    self.update_progress(step + 1, len(timesteps))
                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in tqdm(range(order, steps + 1), disable=os.getenv("DPM_TQDM", "False") == "True"):
                    t = timesteps[step]
                    # We only use lower order for steps < 10
                    # if lower_order_final and steps < 10:
                    if lower_order_final:  # recommended by Shuchen Xue
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(
                        x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type
                    )
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, t)
                    # update progress bar
                    self.update_progress(step + 1, len(timesteps))
            elif method in ["singlestep", "singlestep_fixed"]:
                if method == "singlestep":
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(
                        steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device
                    )
                elif method == "singlestep_fixed":
                    K = steps // order
                    orders = [
                        order,
                    ] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
                for step, order in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    timesteps_inner = self.get_time_steps(
                        skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device
                    )
                    lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                    h = lambda_inner[-1] - lambda_inner[0]
                    r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                    r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                    x = self.singlestep_dpm_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    self.update_progress(step + 1, len(timesteps_outer))
            else:
                raise ValueError(f"Got wrong method {method}")
            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x



    # def sample(
    #     self,
    #     x,
    #     steps=20,
    #     t_start=None,
    #     t_end=None,
    #     order=2,
    #     skip_type="time_uniform",
    #     method="multistep",
    #     lower_order_final=True,
    #     denoise_to_zero=False,
    #     solver_type="dpmsolver",
    #     atol=0.0078,
    #     rtol=0.05,
    #     return_intermediate=False,
    #     flow_shift=1.0,
    # ):
    #     return self.model_fn(x, t) 