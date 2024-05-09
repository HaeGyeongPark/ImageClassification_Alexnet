from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.add_state('true_positive', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_positive', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_negative', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape, "Predictions and targets must have the same shape"
        for cls in range(self.num_classes):
            tp = torch.sum((preds == cls) & (target == cls))
            fp = torch.sum((preds == cls) & (target != cls))
            fn = torch.sum((preds != cls) & (target == cls))

            self.true_positive += tp
            self.false_positive += fp
            self.false_negative += fn

    def compute(self):
        precision = self.true_positive.float() / (self.true_positive.float() + +self.false_positive.float())
        recall = self.true_positive.float() / (self.true_positive.float() + self.false_negative.float())
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_score[torch.isnan(f1_score)] = 0

        return f1_score

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence


        # [TODO] check if preds and target have equal shape


        # [TODO] Cound the number of correct prediction


        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
