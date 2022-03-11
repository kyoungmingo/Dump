import time

import torch
from torch.utils.tensorboard import SummaryWriter

from data import make_data_loader
from losses import make_loss
from models import build_model
from solver import make_optimizer, WarmupMultiStepLR, ClassEvaluator
from utils.common import AverageMeter


def train(args):
    train_loader, val_loader, num_classes, class_types = make_data_loader(args)

    model = build_model(args, num_classes)
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    loss_fn = make_loss(args, num_classes)
    optimizer = make_optimizer(args, model)

    if args.cuda:
        model = model.cuda()

        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.cuda()

    model_state_dict = model.state_dict()
    optim_state_dict = optimizer.state_dict()

    start_epoch = global_step = 0
    if args.pretrained != "":
        weights = torch.load(args.pretrained)
        model_state_dict = weights["state_dict"]
        model.load_state_dict(model_state_dict)

        if args.resume:
            start_epoch = weights["epoch"]
            global_step = weights["global_step"]

            optimizer.load_state_dict(
                torch.load(args.pretrained.replace("model", "optimizer"))["state_dict"]
            )

    scheduler = WarmupMultiStepLR(
        optimizer,
        args.step,
        args.gamma,
        args.warmup_factor,
        args.warmup_step,
        "linear",
        -1 if start_epoch == 0 else start_epoch,
    )

    model_ckpt_dir = args.output_dir / "ckpts"
    model_ckpt_dir.mkdir(exist_ok=True)

    def save_weights(file_name, eph, steps):
        file_name = str(file_name)
        torch.save(
            {"state_dict": model_state_dict, "epoch": eph + 1, "global_step": steps},
            file_name,
        )
        torch.save(
            {"state_dict": optim_state_dict}, file_name.replace("model", "optimizer")
        )

    summary_writer = SummaryWriter(
        log_dir=(args.output_dir / "tensorboard_log"), purge_step=global_step
    )
    print("Training Start!!")
    print(f"Start epochg: {start_epoch}, Global step: {global_step}...")

    evaluator = ClassEvaluator(args, model, num_classes)
    best = {"epoch": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
    batch_time = AverageMeter()
    total_losses = AverageMeter()
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        time_ref = time.time()

        for i, (input_vids, labels) in enumerate(train_loader):
            if args.cuda:
                input_vids = input_vids.cuda()
                labels = labels.cuda()

            # inference & calculate loss
            cls_score = model(input_vids)
            total_loss = loss_fn(cls_score, labels)["X-entropy"]

            summary_writer.add_scalar("Train/cross-entorpy", total_loss, global_step)

            # back propagation
            optimizer.zero_grad()
            if args.debug:
                with torch.autograd.detect_anomaly():
                    total_loss.backward()
            else:
                total_loss.backward()

            optimizer.step()

            batch_time.update(time.time() - time_ref)
            total_losses.update(total_loss.item())

            # learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            summary_writer.add_scalar("Learning rate", current_lr, global_step)

            if (i + 1) % args.log_period == 0:
                print(
                    f"Epoch: [{epoch}][{i+1}/{len(train_loader)}]  "
                    + f"Batch Time {batch_time.val:.3f} ({batch_time.mean:.3f})  "
                    + f"Total_loss {total_losses.val:.3f} ({total_losses.mean:.3f})"
                )

            time_ref = time.time()
            global_step += 1

        print(
            f"Epoch: [{epoch}]\tEpoch Time {batch_time.sum:.3f} s"
            + f"\tLoss {total_losses.mean:.3f}\tLr {current_lr:.2e}"
        )

        # Finish epoch
        batch_time.reset()
        total_losses.reset()
        torch.cuda.empty_cache()

        # Update learning rate
        scheduler.step()

        if (epoch + 1) % args.eval_period == 0 or (epoch + 1) == args.max_epoch:
            print("Calcaulate validation metrics...")
            val_metrics, val_scores, val_labels = evaluator.evaluate(val_loader)

            for idx in range(num_classes):
                summary_writer.add_pr_curve(
                    f"Val/PR_curve/{class_types[idx]}",
                    val_labels == idx,
                    val_scores[:, idx],
                    epoch,
                )

            val_acc = val_metrics["accuracy"]
            val_pre = val_metrics["precision"]
            val_rec = val_metrics["recall"]
            val_f1 = val_metrics["f1_score"]

            # Accuracy
            print(f"Test Accuracy: {val_acc:.2%}")
            summary_writer.add_scalar("Val/accuracy", val_acc, epoch)
            # Other metrics
            print("|   Name   | Precision | Recall | F1 score |")
            print("|------------------------------------------|")

            for idx, cls_type in enumerate(class_types):
                print(
                    f"| {cls_type:^8s} |   {val_pre[idx]:.4f}  |"
                    + f" {val_rec[idx]:.4f} |  {val_f1[idx]:.4f}  |"
                )

                summary_writer.add_scalars(
                    f"Val/{cls_type}",
                    {
                        metric_types: val_metrics[metric_types][idx].data
                        for metric_types in ["precision", "recall", "f1_score"]
                    },
                    epoch,
                )
            print("|------------------------------------------|")
            print(
                f"|   Mean   |   {val_pre.mean():.4f}  "
                + f"| {val_rec.mean():.4f} |  {val_f1.mean():.4f}  |"
            )

            is_best = val_metrics["f1_score"][0] > best["f1_score"]
            if is_best:
                best["epoch"] = epoch
                best["accuracy"] = val_metrics["accuracy"]
                best["precision"] = val_pre.mean().data
                best["recall"] = val_rec.mean().data
                best["f1_score"] = val_f1.mean().data

            if (epoch + 1) % args.save_period == 0 or (epoch + 1) == args.max_epoch:
                pth_file_name = model_ckpt_dir / f"model_{epoch + 1}.pth.tar"
                save_weights(pth_file_name, eph=epoch, steps=global_step)

            if is_best:
                pth_file_name = model_ckpt_dir / "model_best.pth.tar"
                save_weights(pth_file_name, eph=epoch, steps=global_step)

    print(f"Best f1_score {best['f1_score']:.1%}, achived at epoch {best['epoch']}")
    summary_writer.add_hparams(
        {
            "dataset_name": args.dataset_name,
            "input_size": args.vid_size[0],
            "sample_num": args.num_sample_frames,
            "base_lr": args.base_lr,
            "optimizer": args.optimizer,
        },
        {
            "Acc": best["accuracy"],
            "Precision": best["precision"],
            "Recall": best["recall"],
            "F1_score": best["f1_score"],
        },
    )
