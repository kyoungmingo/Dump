import torch


class ClassEvaluator:
    def __init__(self, args, model, num_classes):
        self.cuda = args.cuda
        self.model = model
        self.num_classes = num_classes

    def evaluate(self, val_loader):
        self.model.eval()

        scores, labels = self._parse_features(val_loader)
        scores = torch.softmax(scores, dim=-1)
        metrics = self._cal_metrics(scores, labels, threshold=0.0)

        return metrics, scores, labels

    @torch.no_grad()
    def _parse_features(self, val_loader):
        all_scores, all_labels = [], []
        for vid, labels in val_loader:
            if self.cuda:
                vid = vid.cuda()
                labels = labels.cuda()

            cls_score = self.model(vid)

            all_scores.append(cls_score)
            all_labels.append(labels)

        return torch.cat(all_scores, dim=0), torch.cat(all_labels, dim=0)

    def _cal_metrics(self, scores, labels, threshold=0.0):
        metrics = {
            "true_positive": [],
            "true_negative": [],
            "false_positive": [],
            "false_negative": [],
        }

        predict = torch.argmax(scores, dim=-1)
        for idx in range(self.num_classes):
            cls_pred = (predict == idx) * (scores[:, idx] > threshold)
            cls_labels = labels == idx

            metrics["true_positive"].append((cls_pred * cls_labels).float().sum().data)
            metrics["true_negative"].append(
                (~cls_pred * ~cls_labels).float().sum().data
            )
            metrics["false_positive"].append(
                (cls_pred * ~cls_labels).float().sum().data
            )
            metrics["false_negative"].append(
                (~cls_pred * cls_labels).float().sum().data
            )

        for k, v in metrics.items():
            metrics[k] = torch.as_tensor(v, device=scores.device)

        # Metrics
        metrics["precision"] = torch.where(
            metrics["true_positive"] > 0.0,
            metrics["true_positive"]
            / (metrics["true_positive"] + metrics["false_positive"]),
            torch.zeros_like(metrics["true_positive"]),
        )

        metrics["recall"] = torch.where(
            metrics["true_positive"] > 0.0,
            metrics["true_positive"]
            / (metrics["true_positive"] + metrics["false_negative"]),
            torch.zeros_like(metrics["true_positive"]),
        )
        metrics["f1_score"] = torch.where(
            torch.logical_or(metrics["precision"] > 0.0, metrics["recall"] > 0.0),
            2
            * (metrics["precision"] * metrics["recall"])
            / (metrics["precision"] + metrics["recall"]),
            torch.zeros_like(metrics["precision"]),
        )

        metrics["accuracy"] = torch.sum(metrics["true_positive"]) / predict.size(0)

        return metrics
