# coding=utf-8
# author=yphacker

import tensorflow as tf
import config
from bert import modeling
from bilstm_crf import BiLstmCrf

# https://blog.csdn.net/luoyexuge/article/details/84728649
class BertLstmNer(object):
    def __init__(self, bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings, init_checkpoint):
        self.bert_config = bert_config
        self.is_training = is_training
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labels = labels
        self.num_labels = num_labels
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.init_checkpoint = init_checkpoint

        model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=self.use_one_hot_embeddings
        )
        # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
        embedding = model.get_sequence_output()
        max_seq_length = embedding.shape[1].value

        used = tf.sign(tf.abs(input_ids))
        lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度

        blstm_crf = BiLstmCrf(embedded_chars=embedding, hidden_unit=config.lstm_size, cell_type=config.cell,
                              num_layers=config.num_layers,
                              droupout_rate=config.droupout_rate, initializers=initializers, num_labels=num_labels,
                              seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)

        (self.total_loss, logits, trans, self.pred_ids) = blstm_crf.add_blstm_crf_layer()

        with tf.name_scope("train_op"):
            self.train_op = tf.train.AdamOptimizer().minimize(self.total_loss)
