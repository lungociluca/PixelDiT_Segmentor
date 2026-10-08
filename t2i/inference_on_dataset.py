from typing import List, Union, Optional, Dict, Callable, Any
import logging
import datetime
import time
import math
from collections import abc
from contextlib import ExitStack

import torch
from torch import nn

from detectron2.evaluation import (
    DatasetEvaluator,
    DatasetEvaluators,
)
from detectron2.utils.comm import get_world_size
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds

import config as local_config

def inference_on_dataset(
    model,
    data_loader,
    evaluator: Union[
        DatasetEvaluator,
        List[DatasetEvaluator],
        None,
    ],
    callbacks: Optional[Dict[str, Callable]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: Optional[float] = 1.0,
    use_amp: bool = True,
    amp_dtype: Optional[torch.dtype] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    loss_key: str = "learnable_token_loss",
    ema: Optional[Any] = None,
):
    """
    Run model on data_loader and evaluate metrics.

    If optimizer is provided, this function becomes a training loop and
    expects the model output to contain `loss_key`.

    Optimization features:
        - Gradient accumulation
        - Automatic mixed precision (AMP)
        - BF16 or FP16
        - FP16 GradScaler
        - Gradient clipping
        - Learning-rate scheduler
        - Non-finite loss protection
        - Non-finite gradient protection
        - EMA updates
        - Proper final partial gradient accumulation
        - Optimizer-step callbacks

    Args:
        model:
            Callable / nn.Module accepting a batch from data_loader.

        data_loader:
            Iterable with a fixed length.

        evaluator:
            Detectron2 evaluator(s).

        callbacks:
            Optional callbacks:
                on_start
                before_inference
                after_inference
                before_optimizer_step
                after_optimizer_step
                on_end

        optimizer:
            Optimizer. If None, runs pure inference/evaluation.

        scheduler:
            Optional learning-rate scheduler.

            IMPORTANT:
            The scheduler is stepped once per optimizer update, not once
            per micro-batch.

        gradient_accumulation_steps:
            Number of micro-batches accumulated before optimizer.step().

        max_grad_norm:
            Maximum gradient norm. Set to None to disable clipping.

        use_amp:
            Enable automatic mixed precision during training.

        amp_dtype:
            AMP dtype. Defaults to BF16 when supported, otherwise FP16.

        scaler:
            Optional GradScaler. If None, one is automatically created
            when FP16 AMP is being used.

        loss_key:
            Key containing the scalar loss in model outputs.

        ema:
            Optional EMA object exposing:
                ema.update(model)

    Returns:
        Evaluator results dictionary.
    """

    # ============================================================
    # Validation
    # ============================================================

    loss_logs = open("loss_logs.txt", "w")
    if gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps must be >= 1"
        )

    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError(
            "max_grad_norm must be > 0 or None"
        )

    callbacks = callbacks or {}

    def callback(name, *args, **kwargs):
        fn = callbacks.get(name)
        if fn is not None:
            return fn(*args, **kwargs)
        return None

    num_devices = get_world_size()
    logger = logging.getLogger(__name__)

    total = len(data_loader)

    if total == 0:
        logger.warning("Data loader is empty.")
        evaluator = (
            DatasetEvaluators([])
            if evaluator is None
            else evaluator
        )
        evaluator.reset()
        results = evaluator.evaluate()
        loss_logs.close()
        return {} if results is None else results

    logger.info(
        "Start inference/training on %d batches",
        total,
    )

    # ============================================================
    # Evaluator setup
    # ============================================================

    if evaluator is None:
        evaluator = DatasetEvaluators([])

    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)

    evaluator.reset()

    # ============================================================
    # Training / AMP configuration
    # ============================================================

    training = optimizer is not None

    device = None

    if isinstance(model, nn.Module):
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    amp_enabled = (
        training
        and use_amp
        and device.type == "cuda"
    )

    if amp_enabled:

        if amp_dtype is None:
            # BF16 is generally preferable when the GPU supports it.
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16

        if amp_dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise ValueError(
                "amp_dtype must be torch.float16 or torch.bfloat16"
            )

    else:
        amp_dtype = None

    # GradScaler is required for FP16, but not BF16.
    using_fp16_scaler = (
        amp_enabled
        and amp_dtype == torch.float16
    )

    if using_fp16_scaler and scaler is None:
        scaler = torch.amp.GradScaler("cuda")

    # ============================================================
    # Timing
    # ============================================================

    num_warmup = min(5, total - 1)

    start_time = time.perf_counter()

    total_data_time = 0.0
    total_compute_time = 0.0
    total_eval_time = 0.0

    start_data_time = time.perf_counter()

    # ============================================================
    # Optimization state
    # ============================================================

    micro_step = 0
    optimizer_step = 0

    accumulated_loss = 0.0
    skipped_steps = 0

    # FIX (per-epoch -> global): total number of micro-batches across the
    # entire run. Used so warmup, ETA, and the last-batch flush are computed
    # over the whole run rather than resetting every epoch. This keeps
    # accumulation windows aligned across epoch boundaries (no forced flush
    # at the end of each epoch) and lets the scheduler step exactly the
    # expected number of times.
    total_iters = total * local_config.epochs

    if training:
        optimizer.zero_grad(set_to_none=True)

    # ============================================================
    # Context setup
    # ============================================================

    with ExitStack() as stack:

        if isinstance(model, nn.Module):

            if training:
                model.train()
            else:
                stack.enter_context(
                    inference_context(model)
                )

        if not training:
            stack.enter_context(torch.no_grad())

        callback("on_start")

        # ========================================================
        # Main loop
        # ========================================================
        for ep in range(local_config.epochs):
            print("ep", ep)
            for idx, inputs in enumerate(data_loader):
                print(idx)
                # FIX: global step index across all epochs. Drives warmup,
                # ETA, and the final flush so they behave correctly for
                # multi-epoch runs.
                global_idx = ep * total + idx

                # ----------------------------------------------------
                # Data loading timing
                # ----------------------------------------------------

                total_data_time += (
                    time.perf_counter()
                    - start_data_time
                )

                # Reset timing after warmup (once, globally).
                if global_idx == num_warmup:
                    start_time = time.perf_counter()

                    total_data_time = 0.0
                    total_compute_time = 0.0
                    total_eval_time = 0.0

                # ----------------------------------------------------
                # Forward pass
                # ----------------------------------------------------

                start_compute_time = time.perf_counter()

                callback(
                    "before_inference",
                    idx=idx,
                    inputs=inputs,
                )

                if amp_enabled:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=amp_dtype,
                    ):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                callback(
                    "after_inference",
                    idx=idx,
                    inputs=inputs,
                    outputs=outputs,
                )

                # ----------------------------------------------------
                # Optimization
                # ----------------------------------------------------

                if training:

                    # ----------------------------------------------
                    # Extract loss
                    # ----------------------------------------------

                    losses = []

                    for output in outputs:

                        if loss_key not in output:
                            raise KeyError(
                                f"Model output does not contain "
                                f"'{loss_key}'. Available keys: "
                                f"{list(output.keys())}"
                            )

                        loss_value = output[loss_key]

                        if not torch.is_tensor(loss_value):
                            raise TypeError(
                                f"'{loss_key}' must be a torch.Tensor, "
                                f"got {type(loss_value)}"
                            )

                        if loss_value.numel() != 1:
                            raise ValueError(
                                f"'{loss_key}' must be scalar, "
                                f"got shape {tuple(loss_value.shape)}"
                            )

                        losses.append(loss_value)

                    if not losses:
                        raise ValueError(
                            "Model returned no outputs containing "
                            f"'{loss_key}'."
                        )

                    # Sum losses from all samples/outputs.
                    loss = torch.stack(
                        [x.float() for x in losses]
                    ).sum()

                    # ----------------------------------------------
                    # Non-finite loss protection
                    # ----------------------------------------------

                    if not torch.isfinite(loss).all():

                        skipped_steps += 1

                        logger.warning(
                            "Non-finite loss detected at "
                            "batch=%d (global=%d). Aborting the "
                            "current accumulation window.",
                            idx,
                            global_idx,
                        )

                        optimizer.zero_grad(
                            set_to_none=True
                        )

                        micro_step = 0

                        # FIX: reset the window loss accumulator too.
                        # Previously only micro_step was reset, so the
                        # next optimizer-step log mixed in losses from the
                        # aborted window and divided by the wrong count.
                        accumulated_loss = 0.0

                        continue

                    accumulated_loss += loss.detach().item()

                    # ----------------------------------------------
                    # Gradient accumulation
                    # ----------------------------------------------

                    loss_for_backward = (
                        loss
                        / gradient_accumulation_steps
                    )

                    # ----------------------------------------------
                    # Backward
                    # ----------------------------------------------
                    # FIX: backward used to be guarded by
                    #   `if not local_config.use_learned_tokens:`
                    # which, when the flag was True, skipped backward()
                    # entirely while still running optimizer.step() below.
                    # That stepped the optimizer on stale/zero grads
                    # (and still applied weight decay / momentum / EMA
                    # decay), corrupting the weights. The loss is computed
                    # and validated here specifically to be optimized, so
                    # backward() now runs unconditionally.
                    #
                    # If `use_learned_tokens` was meant to select a
                    # *different* backward path (e.g. backprop only into
                    # the learnable tokens), reinstate that branch here
                    # rather than skipping backward altogether.
                    if using_fp16_scaler:

                        scaler.scale(
                            loss_for_backward
                        ).backward()

                    else:

                        loss_for_backward.backward()

                    micro_step += 1

                    # ------------------------------------------------
                    # Determine whether to update
                    # ------------------------------------------------

                    is_accumulation_boundary = (
                        micro_step
                        % gradient_accumulation_steps
                        == 0
                    )

                    # FIX: only the very last micro-batch of the whole run
                    # (not of each epoch) forces a flush. This keeps
                    # accumulation windows contiguous across epoch
                    # boundaries and avoids an extra scheduler step per
                    # epoch.
                    is_last_batch = (
                        global_idx == total_iters - 1
                    )

                    should_step = (
                        is_accumulation_boundary
                        or is_last_batch
                    )

                    if should_step:
                        print("loss", round(accumulated_loss, 4))
                        callback(
                            "before_optimizer_step",
                            optimizer_step=optimizer_step,
                            batch_idx=idx,
                        )

                        # --------------------------------------------
                        # Unscale FP16 gradients before clipping
                        # --------------------------------------------

                        if using_fp16_scaler:
                            scaler.unscale_(optimizer)

                        # --------------------------------------------
                        # Correct final partial accumulation
                        # --------------------------------------------
                        # Each micro-batch loss is divided by the configured
                        # accumulation length above. When the run ends with
                        # fewer than `gradient_accumulation_steps` valid
                        # micro-batches, those gradients would otherwise be
                        # under-scaled. Rescale them to the actual number of
                        # micro-batches in this optimizer update before
                        # clipping/stepping.
                        if micro_step < gradient_accumulation_steps:
                            correction = (
                                gradient_accumulation_steps / micro_step
                            )
                            for parameter in model.parameters():
                                if parameter.grad is not None:
                                    parameter.grad.mul_(correction)

                        # --------------------------------------------
                        # Gradient clipping
                        # --------------------------------------------

                        grad_norm = None

                        if max_grad_norm is not None:

                            grad_norm = (
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    max_norm=max_grad_norm,
                                )
                            )

                            # grad_norm can be Inf/NaN if gradients
                            # became invalid.
                            if not torch.isfinite(
                                grad_norm
                            ):

                                skipped_steps += 1

                                logger.warning(
                                    "Non-finite gradient norm "
                                    "at batch=%d (global=%d). "
                                    "Skipping optimizer update.",
                                    idx,
                                    global_idx,
                                )

                                # FIX: in FP16, unscale_() has already run
                                # for this optimizer, so the scaler is in a
                                # per-optimizer "unscaled" state. We must
                                # call update() (without step()) to clear
                                # that state and reduce the scale; otherwise
                                # the next unscale_() raises
                                # "unscale_() has already been called ...
                                # since the last update()".
                                if using_fp16_scaler:
                                    scaler.update()

                                optimizer.zero_grad(
                                    set_to_none=True
                                )

                                micro_step = 0

                                # FIX: reset window loss accumulator
                                # (same reason as the non-finite loss path).
                                accumulated_loss = 0.0

                                continue

                        # --------------------------------------------
                        # Optimizer step
                        # --------------------------------------------

                        if using_fp16_scaler:

                            old_scale = scaler.get_scale()

                            scaler.step(optimizer)
                            scaler.update()

                            new_scale = scaler.get_scale()

                            # If the scaler dropped, the optimizer
                            # step was skipped because of Inf/NaN.
                            optimizer_step_was_skipped = (
                                new_scale < old_scale
                            )

                        else:

                            optimizer.step()

                            optimizer_step_was_skipped = False

                        # --------------------------------------------
                        # Scheduler
                        #
                        # IMPORTANT:
                        # Scheduler advances once per optimizer step.
                        # --------------------------------------------

                        if (
                            scheduler is not None
                            and not optimizer_step_was_skipped
                        ):
                            scheduler.step()

                        # --------------------------------------------
                        # EMA
                        # --------------------------------------------

                        if (
                            ema is not None
                            and not optimizer_step_was_skipped
                        ):
                            ema.update(model)

                        # --------------------------------------------
                        # Diagnostics
                        # --------------------------------------------

                        current_lr = max(
                            group["lr"]
                            for group in optimizer.param_groups
                        )

                        callback(
                            "after_optimizer_step",
                            optimizer_step=optimizer_step,
                            batch_idx=idx,
                            loss=(
                                accumulated_loss
                                / max(
                                    1,
                                    min(
                                        gradient_accumulation_steps,
                                        micro_step,
                                    ),
                                )
                            ),
                            lr=current_lr,
                            grad_norm=(
                                None
                                if grad_norm is None
                                else float(grad_norm)
                            ),
                            skipped=optimizer_step_was_skipped,
                        )

                        if not optimizer_step_was_skipped:
                            optimizer_step += 1

                        loss_logs.write(f"{accumulated_loss}\n")

                        # --------------------------------------------
                        # Clear gradients
                        # --------------------------------------------
                        # FIX: moved zero_grad to AFTER the diagnostics /
                        # loss log so the reported window loss and grad_norm
                        # correspond to the step that was just taken. (The
                        # previous ordering zeroed grads, then still reported
                        # grad_norm from the stale pre-step value.)
                        optimizer.zero_grad(
                            set_to_none=True
                        )

                        micro_step = 0
                        accumulated_loss = 0.0

                # ----------------------------------------------------
                # CUDA synchronization for accurate timing
                # ----------------------------------------------------

                if device.type == "cuda":
                    torch.cuda.synchronize()

                total_compute_time += (
                    time.perf_counter()
                    - start_compute_time
                )

                # ----------------------------------------------------
                # Evaluator
                # ----------------------------------------------------
                # FIX: during training the model is in train() mode and
                # returns loss dicts, which a detection evaluator cannot
                # consume. Only feed the evaluator during pure inference.
                start_eval_time = time.perf_counter()

                if not training:
                    evaluator.process(
                        inputs,
                        outputs,
                    )

                total_eval_time += (
                    time.perf_counter()
                    - start_eval_time
                )

                # ----------------------------------------------------
                # Progress logging
                # ----------------------------------------------------

                iters_after_start = (
                    global_idx
                    + 1
                    - num_warmup * int(global_idx >= num_warmup)
                )

                iters_after_start = max(
                    1,
                    iters_after_start,
                )

                data_seconds_per_iter = (
                    total_data_time
                    / iters_after_start
                )

                compute_seconds_per_iter = (
                    total_compute_time
                    / iters_after_start
                )

                eval_seconds_per_iter = (
                    total_eval_time
                    / iters_after_start
                )

                total_seconds_per_iter = (
                    time.perf_counter()
                    - start_time
                ) / iters_after_start

                if (
                    global_idx >= num_warmup * 2
                    or compute_seconds_per_iter > 5
                ):

                    eta = datetime.timedelta(
                        seconds=int(
                            total_seconds_per_iter
                            * (total_iters - global_idx - 1)
                        )
                    )

                    if training:

                        current_lr = max(
                            group["lr"]
                            for group in optimizer.param_groups
                        )

                        log_message = (
                            f"Training done "
                            f"{global_idx + 1}/{total_iters} "
                            f"(epoch {ep + 1}/{local_config.epochs}). "
                            f"Dataloading: "
                            f"{data_seconds_per_iter:.4f} s/iter. "
                            f"Compute: "
                            f"{compute_seconds_per_iter:.4f} s/iter. "
                            f"Eval: "
                            f"{eval_seconds_per_iter:.4f} s/iter. "
                            f"LR: {current_lr:.3e}. "
                            f"Optimizer steps: "
                            f"{optimizer_step}. "
                            f"Skipped: {skipped_steps}. "
                            f"ETA={eta}"
                        )

                    else:

                        log_message = (
                            f"Inference done "
                            f"{global_idx + 1}/{total_iters}. "
                            f"Dataloading: "
                            f"{data_seconds_per_iter:.4f} s/iter. "
                            f"Inference: "
                            f"{compute_seconds_per_iter:.4f} s/iter. "
                            f"Eval: "
                            f"{eval_seconds_per_iter:.4f} s/iter. "
                            f"Total: "
                            f"{total_seconds_per_iter:.4f} s/iter. "
                            f"ETA={eta}"
                        )

                    log_every_n_seconds(
                        logging.INFO,
                        log_message,
                        n=5,
                    )

                start_data_time = time.perf_counter()

        # ========================================================
        # End callback
        # ========================================================

        callback(
            "on_end",
            optimizer_steps=optimizer_step,
            skipped_steps=skipped_steps,
        )

    # ============================================================
    # Final timing
    # ============================================================

    total_time = (
        time.perf_counter()
        - start_time
    )

    total_time_str = str(
        datetime.timedelta(
            seconds=total_time
        )
    )

    # FIX: measured iterations span the whole run, not a single epoch.
    measured_iters = max(
        1,
        total_iters - num_warmup,
    )

    # NOTE:
    # These log formats are intentionally kept compatible with
    # Detectron2 tooling that parses "Total inference time".
    logger.info(
        "Total inference time: {} "
        "({:.6f} s / iter per device, "
        "on {} devices)".format(
            total_time_str,
            total_time / measured_iters,
            num_devices,
        )
    )

    total_compute_time_str = str(
        datetime.timedelta(
            seconds=int(total_compute_time)
        )
    )

    logger.info(
        "Total inference pure compute time: {} "
        "({:.6f} s / iter per device, "
        "on {} devices)".format(
            total_compute_time_str,
            total_compute_time / measured_iters,
            num_devices,
        )
    )

    # ============================================================
    # Training summary
    # ============================================================

    if training:

        logger.info(
            "Optimization summary: "
            "optimizer_steps=%d, skipped_steps=%d, "
            "gradient_accumulation_steps=%d, "
            "amp=%s, amp_dtype=%s",
            optimizer_step,
            skipped_steps,
            gradient_accumulation_steps,
            amp_enabled,
            str(amp_dtype),
        )

    # ============================================================
    # Evaluation
    # ============================================================
    # FIX: only evaluate when running pure inference. During training the
    # evaluator was never fed (see above) and the model is in train() mode,
    # so calling evaluate() here would run forward passes with dropout/BN
    # active and produce meaningless metrics.
    if training:
        results = {}
    else:
        results = evaluator.evaluate()

        # An evaluator may return None on non-main processes.
        if results is None:
            results = {}

    loss_logs.close()
    return results