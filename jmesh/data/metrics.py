import numpy as np
import jittor as jt

def get_confusionmatrix(predicted, target):
    """Computes the confusion matrix
    The shape of the confusion matrix is K x K, where K is the number
    of classes.
    Keyword arguments:
    - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
    predicted scores obtained from the model for N examples and K classes,
    or an N-tensor/array of integer values between 0 and K-1.
    - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
    ground-truth classes for N examples and K classes, or an N-tensor/array
    of integer values between 0 and K-1.
    """
    # If target and/or predicted are tensors, convert them to numpy arrays
    if isinstance(predicted,jt.Var):
        predicted = predicted.numpy()
    if isinstance(target,jt.Var):
        target = target.numpy()

    num_classes = predicted.shape[-2]
    predicted = predicted.argmax(-2)

    predicted = predicted.reshape(-1)
    target = target.reshape(-1)

    

    assert predicted.shape[0] == target.shape[0], \
        'number of targets and predicted outputs do not match'
    
    # # filter -1 
    # valid = target>=0
    # target = target[valid]
    # predicted = predicted[valid]
    if target.shape[0]==0:
        return np.zeros((num_classes,num_classes),dtype=np.int32)

    assert (predicted.max() < num_classes) and (predicted.min() >= 0), \
        'predicted values are not between 0 and k-1'

    assert (target.max() < num_classes) and (target.min() >= 0), \
        'target values are not between 0 and k-1'

   


    

    # hack for bincounting 2 arrays together
    x = predicted + num_classes * target
    bincount_2d = np.bincount(
        x.astype(np.int32), minlength=num_classes**2)
    assert bincount_2d.size == num_classes**2
    conf = bincount_2d.reshape((num_classes, num_classes))
    return conf

def get_iou(conf_matrix,ignore_index=None):
    """Computes the IoU and mean IoU.
    The mean computation ignores NaN elements of the IoU array.
    Returns:
        Tuple: (IoU, mIoU). The first output is the per class IoU,
        for K classes it's numpy.ndarray with K elements. The second output,
        is the mean IoU.
    """
    if ignore_index is not None:
        conf_matrix[:, ignore_index] = 0
        conf_matrix[ignore_index, :] = 0
    true_positive = np.diag(conf_matrix)
    false_positive = np.sum(conf_matrix, 0) - true_positive
    false_negative = np.sum(conf_matrix, 1) - true_positive

    iou = true_positive / np.maximum(1,(true_positive + false_positive + false_negative))
    precision = true_positive / np.maximum(1,(true_positive + false_negative))

    overall_precision = np.sum(true_positive) / np.maximum(1,np.sum(conf_matrix))
    acc = true_positive / np.maximum(1,np.sum(conf_matrix,1))
    
    assert ignore_index == (0,),f"{ignore_index}"
    # valid_classes = np.sum(conf_matrix,1)>0
    miou = np.mean(iou[1:])
    mpre = np.mean(precision[1:])
    macc = np.mean(acc[1:])

    return {
        'iou': iou,
        'acc':acc,
        'mean_iou': miou,
        'mean_acc': macc,
        # 'precision_per_class': precision,
        'mean_precision': mpre,
        'overall_precision': overall_precision
    }

classes = ['unannotated', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain',
               'toilet', 'sink', 'bathtub', 'otherfurniture','ceil']

class IoU:
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).
    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:
        IoU = true_positive / (true_positive + false_positive + false_negative).
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, ignore_index=None,mode="val"):
        self.conf = 0
        self.mode = mode

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")
    
    def add(self, predicted, target):
        conf = get_confusionmatrix(predicted,target)
        self.conf = self.conf+conf
        data = get_iou(conf,self.ignore_index)
        data.pop('iou')
        data.pop('acc')
        data = {f'{self.mode}/batch_{k}':d for k,d in data.items()}
        return data

    def value(self):
        data = get_iou(self.conf,self.ignore_index)
        iou = data.pop('iou')
        acc = data.pop('acc')
        data1 = {f'{self.mode}_iou/{k}':i for k,i in zip(classes,iou)}
        data2 = {f'{self.mode}_acc/{k}':i for k,i in zip(classes,acc)}
        data = {f'{self.mode}/{k}':d for k,d in data.items()}
        data.update(data1) 
        data.update(data2)
        return data

class Accuracy:
    def __init__(self,mode="val"):
        self.n_correct = 0
        self.n_samples = 0
        self.mode = mode

    def add(self,outputs,target):
        if isinstance(outputs,jt.Var):
            outputs = outputs.numpy()
        if isinstance(target,jt.Var):
            target = target.numpy()
        if not isinstance(target,np.ndarray):
            batch_size = 1
            preds = np.argmax(outputs)
        else:
            batch_size = target.shape[0]
            preds = np.argmax(outputs,1)
        correct = (target==preds)
        if len(preds.shape)==2:
            correct = np.array([c[t>=0].mean() for c,t in zip(correct,target)])
            # correct = correct.mean(axis=1)            
        correct = correct.sum()
        self.n_correct += correct.item()
        self.n_samples += batch_size
        return {f"{self.mode}/batch_acc":correct/batch_size}

    def value(self,):
        acc = self.n_correct/max(1,self.n_samples)
        return {f"{self.mode}/acc":acc}