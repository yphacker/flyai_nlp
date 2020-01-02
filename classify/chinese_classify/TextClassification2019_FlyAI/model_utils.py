# coding=utf-8
# author=yphacker

import tensorflow as tf


# def focal_loss(y_pred, y_true, alpha=0.25, gamma=2):
#     r"""Compute focal loss for predictions.
#         Multi-labels Focal loss formula:
#             FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
#                  ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
#     Args:
#      pred: A float tensor of shape [batch_size, num_anchors,
#         num_classes] representing the predicted logits for each class
#      y: A float tensor of shape [batch_size, num_anchors,
#         num_classes] representing one-hot encoded classification targets
#      alpha: A scalar tensor for focal loss alpha hyper-parameter
#      gamma: A scalar tensor for focal loss gamma hyper-parameter
#     Returns:
#         loss: A (scalar) tensor representing the value of the loss function
#     """
#     zeros = tf.zeros_like(y_pred, dtype=y_pred.dtype)
#
#     # For positive prediction, only need consider front part loss, back part is 0;
#     # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
#     pos_p_sub = tf.where(y_true > zeros, y_true - y_pred, zeros)  # positive sample 寻找正样本，并进行填充
#
#     # For negative prediction, only need consider back part loss, front part is 0;
#     # target_tensor > zeros <=> z=1, so negative coefficient = 0.
#     neg_p_sub = tf.where(y_true > zeros, zeros, y_pred)  # negative sample 寻找负样本，并进行填充
#     per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(y_pred, 1e-8, 1.0)) \
#                           - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - y_pred, 1e-8, 1.0))
#
#     return tf.reduce_sum(per_entry_cross_ent)

# classes_num contains sample number of each classes
def focal_loss(prediction_tensor, target_tensor, classes_num=5, alpha=.25, gamma=2., e=0.1):
    '''
    prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
    target_tensor is the label tensor, same shape as predcition_tensor
    '''
    import tensorflow as tf
    from tensorflow.python.ops import array_ops
    from keras import backend as K

    # 1# get focal loss with no balanced weight which presented in paper function (4)
    zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
    one_minus_p = array_ops.where(tf.greater(target_tensor, zeros), target_tensor - prediction_tensor, zeros)
    FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

    # 2# get balanced weight alpha
    classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

    total_num = float(sum(classes_num))
    classes_w_t1 = [total_num / ff for ff in classes_num]
    sum_ = sum(classes_w_t1)
    classes_w_t2 = [ff / sum_ for ff in classes_w_t1]  # scale
    classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
    classes_weight += classes_w_tensor

    alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

    # 3# get balanced focal loss
    balanced_fl = alpha * FT
    balanced_fl = tf.reduce_mean(balanced_fl)

    # 4# add other op to prevent overfit
    # reference : https://spaces.ac.cn/archives/4493
    nb_classes = len(classes_num)
    fianal_loss = (1 - e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor) / nb_classes,
                                                                         prediction_tensor)

    return fianal_loss
