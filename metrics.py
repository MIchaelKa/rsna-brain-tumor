import numpy as np

# do we need? convert to numpy arrray will be better?
import torch

# from sklearn.metrics import confusion_matrix

class AccuracyMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.predictions = []
        self.ground_truth = []
        self.total_correct_samples = 0
        self.total_samples = 0

    def get_predictions(self, output):
        y_pred = (output > 0)#.float()

        # y_pred = torch.sigmoid(output)
        # y_pred = y_pred > 0.5

        return y_pred

    def update(self, y_batch, output):
        y_pred = self.get_predictions(output)
        correct_samples = torch.sum(y_pred == y_batch)
        
        # this is not the history of avg values
        # instead this is history of acc from single batches
        accuracy = float(correct_samples) / y_batch.shape[0]
        self.history.append(accuracy) 

        self.total_correct_samples += correct_samples
        self.total_samples += y_batch.shape[0]

        # Save data for calculating of confusion matrix
        self.predictions.append(y_pred)
        self.ground_truth.append(y_batch)

    def compute_score(self):
        return float(self.total_correct_samples) / self.total_samples

    # def compute_cm(self):
    #     predictions = torch.cat(self.predictions).cpu().numpy()
    #     ground_truth = torch.cat(self.ground_truth).cpu().numpy()

    #     return confusion_matrix(ground_truth, predictions)

class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.total_sum = 0
        self.total_count = 0

    def update(self, x):
        self.history.append(x)
        self.total_sum += x
        self.total_count += 1

    def compute_average(self):
        return np.mean(self.history)

    def moving_average(self, alpha):
        avg_history = [self.history[0]]
        for i in range(1, len(self.history)):
            moving_avg = alpha * avg_history[-1] + (1 - alpha) * self.history[i]
            avg_history.append(moving_avg)
        return avg_history