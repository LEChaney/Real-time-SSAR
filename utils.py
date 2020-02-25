import csv
import pdb
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

class Queue:
    #Constructor creates a list
    def __init__(self, max_size, n_classes):
        self.queue = list(np.zeros((max_size, n_classes),dtype = float).tolist())
        self.max_size = max_size
        self.median = None
        self.ma = None
        self.ewma = None
    #Adding elements to queue
    def enqueue(self,data):
        self.queue.insert(0,data)
        self.median = self._median()
        self.ma = self._ma()
        self.ewma = self._ewma()
        return True

    #Removing the last element from the queue
    def dequeue(self):
        if len(self.queue)>0:
            return self.queue.pop()
        return ("Queue Empty!")

    #Getting the size of the queue
    def size(self):
        return len(self.queue)

    #printing the elements of the queue
    def printQueue(self):
        return self.queue

    #Average   
    def _ma(self):
        return np.array(self.queue[:self.max_size]).mean(axis = 0)

    #Median
    def _median(self):
        return np.median(np.array(self.queue[:self.max_size]), axis = 0)
    
    #Exponential average
    def _ewma(self):
        weights = np.exp(np.linspace(-1., 0., self.max_size))
        weights /= weights.sum()
        average = weights.reshape(1,self.max_size).dot( np.array(self.queue[:self.max_size]))
        return average.reshape(average.shape[1],)

def LevenshteinDistance(a,b):
    # This is a straightforward implementation of a well-known algorithm, and thus
    # probably shouldn't be covered by copyright to begin with. But in case it is,
    # the author (Magnus Lie Hetland) has, to the extent possible under law,
    # dedicated all copyright and related and neighboring rights to this software
    # to the public domain worldwide, by distributing it under the CC0 license,
    # version 1.0. This software is distributed without any warranty. For more
    # information, see <http://creativecommons.org/publicdomain/zero/1.0>
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n
        
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
            
    return current[n]


def LevenshteinDistancePlusAvgFramesEarly(a,b,true_end_frames,detection_frames):
    # This is a straightforward implementation of a well-known algorithm, and thus
    # probably shouldn't be covered by copyright to begin with. But in case it is,
    # the author (Magnus Lie Hetland) has, to the extent possible under law,
    # dedicated all copyright and related and neighboring rights to this software
    # to the public domain worldwide, by distributing it under the CC0 license,
    # version 1.0. This software is distributed without any warranty. For more
    # information, see <http://creativecommons.org/publicdomain/zero/1.0>
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n
        true_end_frames, detection_frames = [-x for x in detection_frames], [-x for x in true_end_frames]
        
    current = range(n+1)
    cur_frames_early = [0]*(n+1)
    cur_matches = [0]*(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        prev_frames_early, cur_frames_early = cur_frames_early, [0]*(n+1)
        prev_matches, cur_matches = cur_matches, [0]*(n+1)
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
            
            cur_frames_early[j] = -np.inf
            if current[j] == add:
                cur_frames_early[j] = max(cur_frames_early[j], prev_frames_early[j])
                cur_matches[j] = prev_matches[j]
            if current[j] == delete:
                cur_frames_early[j] = max(cur_frames_early[j], cur_frames_early[j-1])
                cur_matches[j] = cur_matches[j-1]
            if current[j] == change:
                delta = true_end_frames[j-1] - detection_frames[i-1]
                frames_early = prev_frames_early[j-1] + delta
                cur_frames_early[j] = max(cur_frames_early[j], frames_early)
                if cur_frames_early[j] == frames_early:
                    cur_matches[j] = prev_matches[j-1] + 1
                
            
    avg_frames_early = cur_frames_early[n] / cur_matches[n]
    return current[n], avg_frames_early


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


def calculate_precision(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    return  precision_score(targets.cpu().view(-1), pred.cpu().view(-1), average = 'macro')


def calculate_recall(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    return  recall_score(targets.cpu().view(-1), pred.cpu().view(-1), average = 'macro')
