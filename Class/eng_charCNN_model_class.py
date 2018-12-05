import numpy as np
import copy
import hgtk
from hgtk.exception import NotHangulException
import pymongo

import os
import tensorflow as tf

current_dir_path = os.path.dirname(os.path.realpath(__file__))


class EngCharCNNWithLSTM:
    def __init__(self, scope):
        self._scope = scope
        # self._eng_scope = kor2eng(self._scope)
        self._eng_scope = "asdf"
        self._model_path = os.path.join(current_dir_path, "..", "save", self._eng_scope, self._eng_scope + ".ckpt")

    def _build_model(self):
        self.__create_placeholder()
        self.__create_model()

    def _build_loss(self):
        self.__create_loss_placeholder()
        self.__create_loss()

    def __create_placeholder(self):
        with tf.variable_scope("input_layer", reuse=tf.AUTO_REUSE):
            self._original_sequence_in_char = tf.placeholder(name="original_sequence_in_char", dtype=tf.int64, shape=(self._setting["batch_size"], 1, self._setting["max_word_len"]))

    def __create_model(self):
        with tf.variable_scope(name_or_scope="embedding", reuse=tf.AUTO_REUSE):
            char_embedding_matrix = tf.get_variable(name="char_embedding_matrix", dtype=tf.float64, shape=(self._setting["char_num"], self._setting["char_dim"]))
            char2vec_input_data = tf.nn.embedding_lookup(char_embedding_matrix, self._original_sequence_in_char)

        with tf.variable_scope(name_or_scope="CNN", reuse=tf.AUTO_REUSE):
            CNN_filter = tf.get_variable(name="CNN_filter1", dtype=tf.float64, shape=(1, self._setting["CNN_window"], self._setting["char_dim"], self._setting["CNN_output_dim"]))
            CNN_output = tf.nn.conv2d(name="charCNN1", input=char2vec_input_data, filter=CNN_filter, strides=(1, 1, 1, 1), padding="SAME")
            for i in range(1, self._setting["CNN_layer_num"]):
                CNN_filter = tf.get_variable(name="CNN_filter" + str(i + 1), dtype=tf.float64, shape=(1, self._setting["CNN_window"], self._setting["CNN_output_dim"], self._setting["CNN_output_dim"]))
                CNN_output = tf.nn.conv2d(name="charCNN" + str(i + 1), input=CNN_output, filter=CNN_filter, strides=(1, 1, 1, 1), padding="SAME")
            CNN_output = tf.reduce_mean(CNN_output, axis=2)

        with tf.variable_scope(name_or_scope="Fully_connected_layer", reuse=tf.AUTO_REUSE):
            CNN_output = tf.squeeze(CNN_output, axis=1)
            W = tf.get_variable(name="W", dtype=tf.float64, shape=(self._setting["label_num"], self._setting["CNN_output_dim"]))
            b = tf.get_variable(name="b", dtype=tf.float64, shape=(1, self._setting["label_num"]))
            fully_connected_output = tf.matmul(CNN_output, W, transpose_b=True) + b

            self._output = fully_connected_output

    def __create_loss_placeholder(self):
        with tf.variable_scope("input_layer", reuse=tf.AUTO_REUSE):
            self._labels = tf.placeholder(name="labels", dtype=tf.float64, shape=(self._setting["batch_size"], self._setting["label_num"]))

    def __create_loss(self):
        with tf.variable_scope("loss_layer", reuse=tf.AUTO_REUSE):
            self._loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._labels, logits=self._output)
            train_function = tf.train.AdagradOptimizer(self._setting["learning_rate"])
            self._train_op = train_function.minimize(loss)

    def test(self, model_setting, data):
        idx2char = ["NULL", "UNKNOWN"] + [chr(i) for i in range(97, 123)]
        idx2label = [i for i in range(len(train_data))]

        setting["label_num"] = len(idx2label)
        setting["char_num"] = len(idx2char)

        kor2eng_collection = pymongo.MongoClient("45.119.146.208", 30100, username="admin", password="adminpwd").LISA_TRAIN.kor2eng
        kor2eng_dict = {doc["kor"]: doc["eng"] for doc in kor2eng_collection.find()}

        feed_dict_base = {
            "input_layer/original_sequence_in_char:0": np.zeros(dtype=np.int64, shape=(setting["batch_size"], 1, setting["max_word_len"])),
            "input_layer/labels:0": np.zeros(dtype=np.float64, shape=(setting["batch_size"], setting["label_num"]))
        }

        def kor2eng(word):
            new_word = ""
            for char in word:
                try:
                    splited_char = hgtk.letter.decompose(char)
                    for component in splited_char:
                        if component != "":
                            new_word += kor2eng_dict[component]
                except NotHangulException:
                    new_word += char

            return new_word

        train_feed_dicts = list()
        for answer, question in enumerate(train_data):
            eng_question = kor2eng(question)
            feed_dict = copy.deepcopy(feed_dict_base)
            for char_idx in range(len(eng_question)):
                feed_dict["input_layer/original_sequence_in_char:0"][0][0][char_idx] = idx2char.index(eng_question[char_idx])
            feed_dict["input_layer/labels:0"][0][answer] = 1
            train_feed_dicts.append(feed_dict)

        test_feed_dicts = list()
        for answer, question in enumerate(test_data):
            eng_question = kor2eng(question)
            feed_dict = copy.deepcopy(feed_dict_base)
            for char_idx in range(len(eng_question)):
                feed_dict["input_layer/original_sequence_in_char:0"][0][0][char_idx] = idx2char.index(eng_question[char_idx])
                feed_dict["input_layer/labels:0"][0][answer] = 1
            test_feed_dicts.append(feed_dict)

        session = tf.Session()
        session.run(tf.global_variables_initializer())

        for epoch in range(setting["epoch"]):
            print(epoch)
            for feed_dict in train_feed_dicts:
                local_loss, _ = session.run([loss, train_op], feed_dict=feed_dict)
                print(local_loss)
            print()

        for idx, feed_dict in enumerate(test_feed_dicts):
            test_result = session.run(output, feed_dict=feed_dict)
            test_result = np.array(test_result[0])
            print(idx, np.argmax(test_result))

        print()
