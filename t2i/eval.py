# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
from collections import OrderedDict
import logging
import os

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

from pix_segmentor import Pix_Segmentor
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import CityscapesInstanceEvaluator, CityscapesSemSegEvaluator, \
    SemSegEvaluator, COCOEvaluator, COCOPanopticEvaluator, DatasetEvaluators, verify_results, \
    PascalVOCDetectionEvaluator

from detectron2.utils.logger import setup_logger
from detectron2.data import build_detection_test_loader


from detectron2.utils.file_io import PathManager
from detectron2.evaluation import (
    DatasetEvaluator,
    # inference_on_dataset,
    print_csv_format,
    verify_results,
)
import numpy as np
from PIL import Image
from inference_on_dataset import inference_on_dataset
import config as local_config

# from cat_seg_conf import add_cat_seg_config

from detectron2.data import MetadataCatalog
from register_pascal_20 import register_all_pascal_voc

class VOCbEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)
            pred[pred >= 20] = 20
            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt = np.array(Image.open(f), dtype=int)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    @classmethod
    def build_model(cls, cfg):
        return Pix_Segmentor()
    
    @classmethod
    def build_optimizer(cls, model):

        # ONLY optimize the learnable token
        token_params = [
            param
            for name, param in model.named_parameters()
            if "learnable_token" in name and param.requires_grad
        ]

        assert len(token_params) > 0, \
            "No trainable learnable_token parameter found."

        # TODO
        optimizer = torch.optim.AdamW(
            token_params,
            lr=3e-2,
            weight_decay=0.05,
        )

        return optimizer

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )

        if evaluator_type == "sem_seg_background":
            evaluator_list.append(
                VOCbEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
        ]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # if evaluator_type == "cityscapes_instance":
        #     assert (
        #         torch.cuda.device_count() >= comm.get_rank()
        #     ), "CityscapesEvaluator currently do not work with multiple machines."
        #     return CityscapesInstanceEvaluator(dataset_name)
        # if evaluator_type == "cityscapes_sem_seg":
        #     assert (
        #         torch.cuda.device_count() >= comm.get_rank()
        #     ), "CityscapesEvaluator currently do not work with multiple machines."
        #     return CityscapesSemSegEvaluator(dataset_name)
        # if evaluator_type == "cityscapes_panoptic_seg":
        #     assert (
        #         torch.cuda.device_count() >= comm.get_rank()
        #     ), "CityscapesEvaluator currently do not work with multiple machines."
        #     evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    # @classmethod
    # def test_with_TTA(cls, cfg, model):
    #     logger = logging.getLogger("detectron2.trainer")
    #     # In the end of training, run an evaluation with TTA.
    #     logger.info("Running inference with test-time augmentation ...")
    #     model = SemanticSegmentorWithTTA(cfg, model)
    #     evaluators = [
    #         cls.build_evaluator(
    #             cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
    #         )
    #         for name in cfg.DATASETS.TEST
    #     ]
    #     res = cls.test(cfg, model, evaluators)
    #     res = OrderedDict({k + "_TTA": v for k, v in res.items()})
    #     return res


class CustomTrainer(Trainer):

    @staticmethod
    def trim_data_loader(data_loader: DataLoader, samples_count: int):
        dataset = data_loader.dataset
        trimmed_dataset = Subset(dataset, list(range(min(samples_count, len(dataset)))))
        trimmed_loader = DataLoader(
            trimmed_dataset,
            batch_size=1,
            shuffle=False,  # Disable shuffle for trimmed data to preserve order
            num_workers=data_loader.num_workers,
            collate_fn=data_loader.collate_fn,
            pin_memory=data_loader.pin_memory,
            drop_last=False,  # Don't drop samples from trimmed data
        )
        return trimmed_loader

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = CustomTrainer.trim_data_loader(cls.build_test_loader(cfg, dataset_name), local_config.eval_samples_limit)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            optimizer = cls.build_optimizer(model)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=math.ceil(len(data_loader) / local_config.grad_accumulation),
                eta_min=1e-4,
            )

            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator,
                optimizer=optimizer, gradient_accumulation_steps=local_config.grad_accumulation)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @staticmethod
    def freeze_pretrained_backbone(model):
        """Freeze the pretrained backbone and leave only the probe token trainable."""
        for name, param in model.named_parameters():
            param.requires_grad = ("learnable_token" in name)
        return model

    @staticmethod
    def train_learnable_token_only(model, dataloader, optimizer, device, epochs=1, max_steps=None, accumulation_steps=1):
        """Train only the learnable token with optional gradient accumulation.

        This matches the intended behavior of a readout-only token: it can attend to text and image
        tokens, but it does not modify the original model's forward behavior.
        """
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1")

        if hasattr(model, "model"):
            backbone = model.model
        else:
            backbone = model

        backbone.train()
        CustomTrainer.freeze_pretrained_backbone(backbone)

        total_loss = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)

        for _ in range(epochs):
            for batch in dataloader:
                if isinstance(batch, dict):
                    if "image" in batch:
                        x = batch["image"].to(device)
                    else:
                        raise KeyError("batch dictionary must contain an 'image' field")
                    if "text_emb" in batch:
                        y = batch["text_emb"].to(device)
                    else:
                        raise KeyError("batch dictionary must contain a 'text_emb' field")
                else:
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)

                t = torch.zeros(x.shape[0], device=device, dtype=x.dtype)

                with torch.set_grad_enabled(True):
                    _ = backbone(x, t, y, s=None, mask=None)

                loss = F.mse_loss(backbone.learnable_token, backbone.learnable_token_target.to(device))
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()

                total_loss += float(loss.item())
                steps += 1

                if steps % accumulation_steps == 0 or (max_steps is not None and steps >= max_steps):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if max_steps is not None and steps >= max_steps:
                    return total_loss / steps

        if steps % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        return total_loss / max(steps, 1)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    # register_all_pascal_voc(os.getenv("DETECTRON2_DATASETS", "datasets"))
    cfg = get_cfg()
    # for poly lr schedule
    # add_cat_seg_config(cfg)
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = local_config.ds_config_path
    cfg.DATASETS.TEST = [local_config.eval_dataset.value]
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")
    return cfg


def main(args):
    cfg = setup(args)
    torch.set_float32_matmul_precision("high")

    model = CustomTrainer.build_model(cfg)
    res = CustomTrainer.test(cfg, model)
    if cfg.TEST.AUG.ENABLED:
        res.update(CustomTrainer.test_with_TTA(cfg, model))
    if comm.is_main_process():
        verify_results(cfg, res)
    return res


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )