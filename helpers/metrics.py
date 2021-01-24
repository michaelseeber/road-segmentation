
import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils import metrics_utils

####losses

@tf.function
def f1(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    return 2 * (tf.keras.backend.sum(y_true * y_pred)+ tf.keras.backend.epsilon()) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + tf.keras.backend.epsilon())

@tf.function
def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@tf.function
def focal_loss_with_logits(targets, logits, alpha=0.25, gamma=2):
    probabilities = tf.math.sigmoid(logits)
    weight_a = alpha * (1 - probabilities) ** gamma * targets
    weight_b = (1 - alpha) * probabilities ** gamma * (1 - targets)

    return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

@tf.function
def soft_dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

@tf.function
def bce_loss(y_true, y_pred):
  #note: do not confuse with balanced cross entropy, which is often referred to as BCE
  loss = losses.binary_crossentropy(y_true, y_pred)
  return loss

@tf.function
def root_mean_squared_error(y_true, y_pred):
  loss = tf.sqrt(losses.mean_squared_error(y_true, y_pred))
  return loss

@tf.function
def lovasz_hinge(labels, logits, per_image=True, ignore=None):
#def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

@tf.function
def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   #strict=True,
                   name="loss"
                   )
    return loss

@tf.function
def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

@tf.function
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

@tf.function
def CE_and_lovasz_hinge(labels, logits):
    CE = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    return 0.5 * CE(labels, logits) + 0.5 * lovasz_hinge(labels, logits)

@tf.function
def BCE_and_lovasz_hinge(labels, logits, batch_size=32, img_height=256, img_width=256):
    beta = tf.reduce_sum(1 - labels) / (batch_size * img_height * img_width) + tf.keras.backend.epsilon()
    pos_weight = beta / (1 - beta)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=logits, labels=labels, pos_weight=pos_weight)

    # or reduce_sum and/or axis=-1
    BCE = tf.reduce_mean(loss * (1 - beta))

    return 0.5 * BCE + 0.5 * lovasz_hinge(labels, logits)

@tf.function
def BCE_and_soft_dice(labels, probabilities, batch_size=32, img_height=256, img_width=256):
    beta = tf.reduce_sum(1 - labels) / (batch_size * img_height * img_width) + tf.keras.backend.epsilon()
    def convert_to_logits(y_pred):
      # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

      return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
      y_pred = convert_to_logits(y_pred)
      pos_weight = beta / (1 - beta)
      loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)
  
      # or reduce_sum and/or axis=-1
      return tf.reduce_mean(loss * (1 - beta))

    #return 0.5 * BCE + 0.5 * soft_dice_loss(labels, probabilities)
    return 0.5 * loss(labels, probabilities) + 0.5 * soft_dice_loss(labels, probabilities)

@tf.function
def BCE_with_logits(labels, logits, batch_size=32, img_height=256, img_width=256):
    beta = tf.reduce_sum(1 - labels) / (batch_size * img_height * img_width) + tf.keras.backend.epsilon()
    pos_weight = beta / (1 - beta)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=logits, labels=labels, pos_weight=pos_weight)

    # or reduce_sum and/or axis=-1
    BCE = tf.reduce_mean(loss * (1 - beta))

    return BCE

####metrics
class meanIoU_with_probabilities(tf.keras.metrics.MeanIoU):
    #NOTE: tf.keras.metrics.MeanIoU does work only with binary predictions :(
    #https://stackoverflow.com/questions/59990587/tf-keras-metrics-meaniou-with-sigmoid-layer
    def update_state(self, y_true, y_pred, sample_weight=None):
        #th = 0.5
        ##y_true_ = tf.compat.v1.to_int32(y_true)
        ##y_pred_ = tf.compat.v1.to_int32(y_pred > th)
        #y_true_ = tf.cast(y_true, tf.int32)
        #y_pred_ = tf.cast(y_pred > th, tf.int32)
        #return super().update_state(y_true_, y_pred_, sample_weight)
        th = 0.5
        y_true = math_ops.cast(y_true, tf.int32)
        y_pred = math_ops.cast(y_pred > th, tf.int32)
        #return super().update_state(y_true, y_pred, sample_weight)
        super().update_state(y_true, y_pred, sample_weight)

class meanIoU_with_logits(tf.keras.metrics.MeanIoU):
    #NOTE: tf.keras.metrics.MeanIoU does work only with binary predictions :(
    #https://stackoverflow.com/questions/59990587/tf-keras-metrics-meaniou-with-sigmoid-layer
    def update_state(self, y_true, y_pred, sample_weight=None):
        #th = 0.0
        ##y_true_ = tf.compat.v1.to_int32(y_true)
        ##y_pred_ = tf.compat.v1.to_int32(y_pred > th)
        #y_true_ = tf.cast(y_true, tf.int32)
        #y_pred_ = tf.cast(y_pred > th, tf.int32)
        #return super().update_state(y_true_, y_pred_, sample_weight)
        th = 0.0
        y_true = math_ops.cast(y_true, tf.int32)
        y_pred = math_ops.cast(y_pred > th, tf.int32)
        #return super().update_state(y_true, y_pred, sample_weight)
        super().update_state(y_true, y_pred, sample_weight)

class Precision_with_logits(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        th = 0.0
        y_pred = math_ops.cast(y_pred > th, self._dtype)
        #return super().update_state(y_true, y_pred, sample_weight)
        super().update_state(y_true, y_pred, sample_weight)

class Recall_with_logits(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        th = 0.0
        y_pred = math_ops.cast(y_pred > th, self._dtype)
        #return super().update_state(y_true, y_pred, sample_weight)
        super().update_state(y_true, y_pred, sample_weight)

#class F1Score_with_probabilities(tf.keras.metrics.Metric):
#    #problem with other F1 score implementation: batch problem, reason why F1 score was removed from TF https://github.com/keras-team/keras/issues/5794
#    def __init__(self, name="f1_score_with_probabilities", **kwargs):
#        super(F1Score_with_probabilities, self).__init__(name=name, **kwargs)
#        self.true_positives = self.add_weight(name="f1_true_positives", initializer="zeros", shape=(1,))
#        self.false_positives = self.add_weight(name="f1_false_positives", initializer="zeros", shape=(1,))
#        self.false_negatives = self.add_weight(name="f1_false_negatives", initializer="zeros", shape=(1,))
#
#    def update_state(self, y_true, y_pred, sample_weight=None):
#        metrics_utils.update_confusion_matrix_variables(
#            {
#                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
#                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
#                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
#            },
#            y_true,
#            y_pred,
#            thresholds=0.5,
#            top_k=None,
#            class_id=None,
#            sample_weight=sample_weight)
#
#    def result(self):
#        precision = math_ops.div_no_nan(self.true_positives, self.true_positives + self.false_positives)
#        recall =    math_ops.div_no_nan(self.true_positives, self.true_positives + self.false_negatives)
#        f1_score =  math_ops.div_no_nan(2.0 * precision * recall, precision + recall)
#        return  f1_score[0]
#
#    def reset_states(self):
#        # The state of the metric will be reset at the start of each epoch.
#        self.true_positives.assign(0.0)
#        self.false_positives.assign(0.0)
#        self.false_negatives.assign(0.0)
#
#class F1Score_with_logits(F1Score_with_probabilities):
#    #problem with other F1 score implementation: batch problem, reason why F1 score was removed from TF https://github.com/keras-team/keras/issues/5794
#    #implemented by combining precision and recall implementation
#    def update_state(self, y_true, y_pred, sample_weight=None):
#        th = 0.0
#        y_pred = math_ops.cast(y_pred > th, self._dtype)
#        super().update_state(y_true, y_pred, sample_weight)

#class F1Score_with_probabilities(tf.keras.metrics.Metric):
#    #problem with other F1 score implementation: batch problem, reason why F1 score was removed from TF https://github.com/keras-team/keras/issues/5794
#    #implemented by combining precision and recall implementation
#    def __init__(self,
#               thresholds=None,
#               #thresholds=0.5,
#               top_k=None,
#               class_id=None,
#               name=None,
#               dtype=None):
#        super(F1Score_with_probabilities, self).__init__(name=name, dtype=dtype)
#        self.init_thresholds = thresholds
#        self.top_k = top_k
#        self.class_id = class_id
#
#        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
#        self.thresholds = metrics_utils.parse_init_thresholds(
#            thresholds, default_threshold=default_threshold)
#        self.true_positives = self.add_weight(
#            'true_positives',
#            shape=(len(self.thresholds),),
#            initializer=init_ops.zeros_initializer)
#        self.false_positives = self.add_weight(
#            'false_positives',
#            shape=(len(self.thresholds),),
#            initializer=init_ops.zeros_initializer)
#        self.false_negatives = self.add_weight(
#            'false_negatives',
#            shape=(len(self.thresholds),),
#            initializer=init_ops.zeros_initializer)
#    
#    def update_state(self, y_true, y_pred, sample_weight=None):
#        #return metrics_utils.update_confusion_matrix_variables(
#        metrics_utils.update_confusion_matrix_variables(
#            {
#                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
#                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
#                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
#            },
#            y_true,
#            y_pred,
#            thresholds=self.thresholds,
#            top_k=self.top_k,
#            class_id=self.class_id,
#            sample_weight=sample_weight)
#    
#    def result(self):
#        precision = math_ops.div_no_nan(self.true_positives, self.true_positives + self.false_positives)
#        #precision = precision[0] if len(self.thresholds) == 1 else precision
#        recall =    math_ops.div_no_nan(self.true_positives, self.true_positives + self.false_negatives)
#        #recall =    recall[0] if len(self.thresholds) == 1 else recall
#        f1_score =  math_ops.div_no_nan(2.0 * precision * recall, precision + recall)
#        return  f1_score[0] if len(self.thresholds) == 1 else f1_score
#
#    def reset_states(self):
#        num_thresholds = len(to_list(self.thresholds))
#        K.batch_set_value(
#            [(v, np.zeros((num_thresholds,))) for v in self.variables])
#
#    def get_config(self):
#        config = {
#            'thresholds': self.init_thresholds,
#            'top_k': self.top_k,
#            'class_id': self.class_id
#        }
#        base_config = super(F1Score_with_probabilities, self).get_config()
#        return dict(list(base_config.items()) + list(config.items()))

#class F1Score_with_logits(F1Score_with_probabilities):
#    #problem with other F1 score implementation: batch problem, reason why F1 score was removed from TF https://github.com/keras-team/keras/issues/5794
#    #implemented by combining precision and recall implementation
#    def update_state(self, y_true, y_pred, sample_weight=None):
#        th = 0.0
#        y_pred = math_ops.cast(y_pred > th, self._dtype)
#        super().update_state(y_true, y_pred, sample_weight)

#class F1Score_with_logits(tf.keras.metrics.Precision):
#    #problem with other F1 score implementation: batch problem, reason why F1 score was removed from TF https://github.com/keras-team/keras/issues/5794
#    #implemented by combining precision and recall implementation
#    def __init__(self,
#               #thresholds=None,
#               thresholds=0.0,
#               top_k=None,
#               class_id=None,
#               name=None,
#               dtype=None):
#        super(F1Score_with_logits, self).__init__(name=name, dtype=dtype)
#        self.init_thresholds = thresholds
#        self.top_k = top_k
#        self.class_id = class_id
#
#        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
#        self.thresholds = metrics_utils.parse_init_thresholds(
#            thresholds, default_threshold=default_threshold)
#        self.true_positives = self.add_weight(
#            'true_positives',
#            shape=(len(self.thresholds),),
#            initializer=init_ops.zeros_initializer)
#        self.false_positives = self.add_weight(
#            'false_positives',
#            shape=(len(self.thresholds),),
#            initializer=init_ops.zeros_initializer)
#        self.false_negatives = self.add_weight(
#            'false_negatives',
#            shape=(len(self.thresholds),),
#            initializer=init_ops.zeros_initializer)
#    def update_state(self, y_true, y_pred, sample_weight=None):
#        y_pred_ = tf.cast(y_pred > self.thresholds, tf.float32)
#        return metrics_utils.update_confusion_matrix_variables(
#            {
#                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
#                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
#                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
#            },
#            y_true,
#            y_pred_,
#            thresholds=0.5,
#            top_k=self.top_k,
#            class_id=self.class_id,
#            sample_weight=sample_weight)
#    def result(self):
#        precision = math_ops.div_no_nan(self.true_positives, self.true_positives + self.false_positives)
#        #precision = precision[0] if len(self.thresholds) == 1 else precision
#        recall =    math_ops.div_no_nan(self.true_positives, self.true_positives + self.false_negatives)
#        #recall =    recall[0] if len(self.thresholds) == 1 else recall
#        f1_score =  math_ops.div_no_nan(2.0 * precision * recall, precision + recall)
#        return  f1_score[0] if len(self.thresholds) == 1 else f1_score
