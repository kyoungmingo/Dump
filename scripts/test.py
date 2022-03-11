import numpy as np

import torch

from data import make_data_loader
from data.datasets import init_dataset
from models import build_model
from solver.evaluator import ClassEvaluator


def test(args):
    dataset = init_dataset(args.dataset_name, root=args.dataset_root, verbose=False)
    _, test_loader, num_classes, class_types = make_data_loader(args)

    model = build_model(args, num_classes)
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    if args.cuda:
        model = model.cuda()

    if args.pretrained != "":
        weights = torch.load(args.pretrained)
        model_state_dict = weights["state_dict"]
        model.load_state_dict(model_state_dict)

    print("Testing Start!!")
    evaluator = ClassEvaluator(args, model, num_classes)

    test_metrics, test_scores, test_labels = evaluator.evaluate(test_loader)

    test_predict = torch.argmax(test_scores, dim=-1)

    val_acc = test_metrics["accuracy"]
    val_pre = test_metrics["precision"]
    val_rec = test_metrics["recall"]
    val_f1 = test_metrics["f1_score"]

    # Accuracy
    print(f"Accuracy: {val_acc:.2%}")
    print("   Name   | Precision | Recall | F1 score |")
    for idx, cls_type in enumerate(class_types):
        print(
            f" {cls_type:^8s} |   {val_pre[idx]:.4f}  | {val_rec[idx]:.4f} |  {val_f1[idx]:.4f}  |"
        )
    print(
        f"   Mean   |   {val_pre.mean():.4f}  | {val_rec.mean():.4f} |  {val_f1.mean():.4f}  |"
    )

    for idx, cls_type in enumerate(class_types):
        print(cls_type)
        print(f"{int(test_metrics['true_positive'][idx])} | {int(test_metrics['false_negative'][idx])}")
        print(f"{int(test_metrics['false_positive'][idx])} | {int(test_metrics['true_negative'][idx])}")

    # test_dataset = dataset.test
    # mis_predicted_list = []
    # for idx, (predict, label) in enumerate(zip(test_predict, test_labels)):
    #     if predict != label:
    #         print(
    #             f"Predict: {class_types[predict]} -> Label: {str(test_dataset[idx][0].name)}"
    #         )
    #         mis_predicted_list.append(test_dataset[idx][0])
    # np.savetxt(
    #     "mis_predicted.csv", np.array(mis_predicted_list), delimiter=",", fmt="%s"
    # )

