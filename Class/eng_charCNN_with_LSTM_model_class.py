import os
import math
import numpy as np
import tensorflow as tf
from copy import deepcopy
from sklearn.model_selection import train_test_split

from Function.InnerFunction.model_function import load_share, save_share, load_train_data
from Function.InnerFunction.get_valid_tag_function import get_valid_tag
from Function.InnerFunction.kor2eng_function import kor2eng

current_dir_path = os.path.dirname(os.path.realpath(__file__))


class EngCharCNNWithLSTM:
    def __init__(self, scope, setting, is_train=False):
        self.__load_setting(scope, setting, is_train)
        self.__graph = tf.Graph()

        if self.__is_train:
            input_data, output_data = load_train_data(self.__scope)
            train_questions, test_questions, train_answers, test_answers = train_test_split(input_data, output_data, test_size=0.2)
            self.__train_data = [{"question": q, "answer": a} for q, a in zip(train_questions, train_answers)]
            self.__test_data = [{"question": q, "answer": a} for q, a in zip(test_questions, test_answers)]
            self.__create_share(input_data, output_data)
            for i in range(self.__setting["max_batch_size"], 0, -1):
                if len(self.__train_data) % i == 0:
                    self.__setting["batch_size"] = i
                    break
        else:
            self.__setting["batch_size"] = 1
            self.__share = load_share(self.__scope)
            if self.__share:
                self.__setting["char_num"] = len(self.__share["idx2char"])
                self.__setting["label_num"] = len(self.__share["idx2label"])

        if self.__model_exist():
            self.__build_model()

        if self.__is_train:
            save_share(self.__scope, self.__share)
            self.__session.run(self.__init)
        else:
            self.__load_model()

    def __load_setting(self, scope, setting, is_train):
        self.__scope = scope
        self.__eng_scope = kor2eng(self.__scope)
        self.__is_train = is_train
        self.__setting = setting
        self.__save_directory = os.path.join(current_dir_path, "..", "save", self.__eng_scope)
        self.__model_path = os.path.join(self.__save_directory, self.__eng_scope + ".ckpt")
        self.__share = load_share(self.__scope)

    def __model_exist(self):
        if self.__is_train:
            if self.__share:
                return True
            else:
                return False
        else:
            if self.__share and os.path.exists(self.__save_directory):
                return True
            else:
                return False

    def __create_share(self, input_data, output_data):
        if input_data and output_data:
            self.__share = dict()
            self.__share["idx2char"], self.__share["idx2label"] = ["NULL", "UNKNOWN"], list()
            for input_data, output_data in zip(input_data, output_data):
                for word in input_data:
                    eng_word = kor2eng(word)
                    for char in eng_word:
                        if char not in self.__share["idx2char"]:
                            self.__share["idx2char"].append(char)

                if output_data not in self.__share["idx2label"]:
                    self.__share["idx2label"].extend(output_data)

            self.__setting["char_num"] = len(self.__share["idx2char"])
            self.__setting["label_num"] = len(self.__share["idx2label"])
        else:
            self.__share = None

    def __save_model(self):
        self.__saver.save(self.__session, self.__model_path)

    def __load_model(self):
        if self.__model_exist():
            self.__saver.restore(self.__session, self.__model_path)

    def __build_model(self):
        with self.__graph.as_default():
            with tf.variable_scope(name_or_scope=self.__eng_scope, reuse=tf.AUTO_REUSE):
                self.__create_placeholder()
                self.__create_model()
                if self.__is_train:
                    self.__create_loss_placeholder()
                    self.__create_loss()
            self.__init = tf.global_variables_initializer()
            self.__saver = tf.train.Saver()

        self.__session = tf.Session(graph=self.__graph)

    def __create_placeholder(self):
        self.__original_sequence_in_char = tf.placeholder(name="original_sequence_in_char", dtype=tf.int64, shape=(self.__setting["batch_size"], self.__setting["max_sentence_len"], self.__setting["max_word_len"]))
        self.__original_sequence_lengths = tf.placeholder(name="original_sequence_lengths", dtype=tf.int64, shape=(self.__setting["batch_size"], ))

    def __create_model(self):
        with tf.variable_scope(name_or_scope="charCNN_layer", reuse=tf.AUTO_REUSE):
            with tf.variable_scope(name_or_scope="embedding", reuse=tf.AUTO_REUSE):
                char_embedding_matrix = tf.get_variable(name="char_embedding_matrix", dtype=tf.float64, shape=(self.__setting["char_num"], self.__setting["char_dim"]))
                char2vec_input_data = tf.nn.embedding_lookup(char_embedding_matrix, self.__original_sequence_in_char)

            with tf.variable_scope(name_or_scope="CNN", reuse=tf.AUTO_REUSE):
                CNN_filter = tf.get_variable(name="CNN_filter1", dtype=tf.float64, shape=(1, self.__setting["CNN_window"], self.__setting["char_dim"], self.__setting["CNN_output_dim"]))
                CNN_output = tf.nn.conv2d(name="charCNN1", input=char2vec_input_data, filter=CNN_filter, strides=(1, 1, 1, 1), padding="SAME")
                for i in range(1, self.__setting["CNN_layer_num"]):
                    CNN_filter = tf.get_variable(name="CNN_filter" + str(i + 1), dtype=tf.float64, shape=(1, self.__setting["CNN_window"], self.__setting["CNN_output_dim"], self.__setting["CNN_output_dim"]))
                    CNN_output = tf.nn.conv2d(name="charCNN" + str(i + 1), input=CNN_output, filter=CNN_filter, strides=(1, 1, 1, 1), padding="SAME")
                CNN_output = tf.reduce_mean(CNN_output, axis=2)

        with tf.variable_scope(name_or_scope="RNN_layer", reuse=tf.AUTO_REUSE):
            with tf.variable_scope(name_or_scope="LSTM_encoding", reuse=tf.AUTO_REUSE):
                cell = tf.nn.rnn_cell.BasicLSTMCell(name="cell1", num_units=self.__setting["LSTM_output_dim"])
                RNN_output, state = tf.nn.dynamic_rnn(cell=cell, inputs=CNN_output, sequence_length=self.__original_sequence_lengths, initial_state=cell.zero_state(batch_size=self.__setting["batch_size"], dtype=tf.float64))
                for i in range(1, self.__setting["RNN_layer_num"]):
                    cell = tf.nn.rnn_cell.BasicLSTMCell(name="cell" + str(i + 1), num_units=self.__setting["LSTM_output_dim"])
                    RNN_output, state = tf.nn.dynamic_rnn(cell=cell, inputs=CNN_output, sequence_length=self.__original_sequence_lengths, initial_state=cell.zero_state(batch_size=self.__setting["batch_size"], dtype=tf.float64))

        with tf.variable_scope(name_or_scope="Fully_connected_layer", reuse=tf.AUTO_REUSE):
            splited = [tf.squeeze(x, axis=0) for x in tf.split(value=RNN_output, num_or_size_splits=self.__setting["batch_size"], axis=0)]
            W = tf.get_variable(name="W", dtype=tf.float64, shape=(self.__setting["label_num"], self.__setting["LSTM_output_dim"]))
            b = tf.get_variable(name="b", dtype=tf.float64, shape=(self.__setting["max_sentence_len"], self.__setting["label_num"]))
            fully_connected_output = list()
            for val in splited:
                result = tf.matmul(val, W, transpose_b=True) + b
                fully_connected_output.append(tf.expand_dims(result, axis=0))
            fully_connected_output = tf.concat(fully_connected_output, axis=0)

        output = tf.reduce_max(fully_connected_output, axis=1)
        self.__output = tf.nn.softmax(output, axis=1)

    def __create_loss_placeholder(self):
        self.__labels = tf.placeholder(name="labels", dtype=tf.float64, shape=(self.__setting["batch_size"], self.__setting["label_num"]))

    def __create_loss(self):
        with tf.variable_scope(name_or_scope="loss_layer", reuse=tf.AUTO_REUSE):
            self.__loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.__labels, logits=self.__output)
            train_function = tf.train.AdagradOptimizer(self.__setting["learning_rate"])
            self.__train_op = train_function.minimize(self.__loss)

    def run(self, msg=None):
        if self.__is_train:
            feed_dict_base = {
                self.__original_sequence_in_char.name: np.zeros(dtype=np.int64, shape=(self.__setting["batch_size"], self.__setting["max_sentence_len"], self.__setting["max_word_len"])),
                self.__original_sequence_lengths.name: np.zeros(dtype=np.int64, shape=(self.__setting["batch_size"], )),
                self.__labels.name: np.zeros(dtype=np.float64, shape=(self.__setting["batch_size"], self.__setting["label_num"]))
            }

            train_steps = math.ceil(len(self.__train_data) / self.__setting["batch_size"])

            train_feed_dicts = [deepcopy(feed_dict_base) for i in range(train_steps)]
            for step_idx in range(train_steps):
                for batch_idx in range(min(self.__setting["batch_size"], len(self.__train_data[self.__setting["batch_size"] * step_idx:]))):
                    for word_idx in range(len(self.__train_data[self.__setting["batch_size"] * step_idx + batch_idx]["question"])):
                        char_word = kor2eng(self.__train_data[self.__setting["batch_size"] * step_idx + batch_idx]["question"][word_idx])
                        for char_idx in range(len(char_word)):
                            if char_word[char_idx] in self.__share["idx2char"]:
                                train_feed_dicts[step_idx][self.__original_sequence_in_char.name][batch_idx][word_idx][char_idx] = self.__share["idx2char"].index(char_word[char_idx])
                            else:
                                train_feed_dicts[step_idx][self.__original_sequence_in_char.name][batch_idx][word_idx][char_idx] = self.__share["idx2char"].index("UNKNOWN")
                    train_feed_dicts[step_idx][self.__original_sequence_lengths.name][batch_idx] = len(self.__train_data[self.__setting["batch_size"] * step_idx + batch_idx]["question"])
                    for answer_idx in range(len(self.__train_data[self.__setting["batch_size"] * step_idx + batch_idx]["answer"])):
                        train_feed_dicts[step_idx][self.__labels.name][batch_idx][self.__share["idx2label"].index(self.__train_data[self.__setting["batch_size"] * step_idx + batch_idx]["answer"][answer_idx])] = 1 / len(self.__train_data[self.__setting["batch_size"] * step_idx + batch_idx]["answer"])

            test_steps = math.ceil(len(self.__test_data) / self.__setting["batch_size"])

            test_feed_dicts = [deepcopy(feed_dict_base) for i in range(test_steps)]
            for step_idx in range(test_steps):
                for batch_idx in range(min(self.__setting["batch_size"], len(self.__test_data[self.__setting["batch_size"] * step_idx:]))):
                    for word_idx in range(len(self.__test_data[self.__setting["batch_size"] * step_idx + batch_idx]["question"])):
                        char_word = kor2eng(self.__test_data[self.__setting["batch_size"] * step_idx + batch_idx]["question"][word_idx])
                        for char_idx in range(len(char_word)):
                            if char_word[char_idx] in self.__share["idx2char"]:
                                test_feed_dicts[step_idx][self.__original_sequence_in_char.name][batch_idx][word_idx][char_idx] = self.__share["idx2char"].index(char_word[char_idx])
                            else:
                                test_feed_dicts[step_idx][self.__original_sequence_in_char.name][batch_idx][word_idx][char_idx] = self.__share["idx2char"].index("UNKNOWN")
                    test_feed_dicts[step_idx][self.__original_sequence_lengths.name][batch_idx] = len(self.__test_data[self.__setting["batch_size"] * step_idx + batch_idx]["question"])
                    for answer_idx in range(len(self.__test_data[self.__setting["batch_size"] * step_idx + batch_idx]["answer"])):
                        test_feed_dicts[step_idx][self.__labels.name][batch_idx][self.__share["idx2label"].index(self.__test_data[self.__setting["batch_size"] * step_idx + batch_idx]["answer"][answer_idx])] = 1 / len(self.__test_data[self.__setting["batch_size"] * step_idx + batch_idx]["answer"])

            for epoch in range(self.__setting["epoch"]):
                losses = list()
                for feed_dict in train_feed_dicts:
                    local_loss, _ = self.__session.run([self.__loss, self.__train_op], feed_dict=feed_dict)
                    local_loss = sum(local_loss.tolist()) / self.__setting["batch_size"]
                    losses.append(local_loss)
                print("EPOCH :", epoch + 1, "/ LOSS :", sum(losses) / len(losses))

            outputs = list()
            for feed_dict in test_feed_dicts:
                output = self.__session.run(self.__output, feed_dict=feed_dict)
                outputs.extend(output.tolist())
            outputs = np.array(outputs)
            results = np.argmax(outputs, axis=1).tolist()
            results = [self.__share["idx2label"][idx] for idx in results]
            correct = 0
            for predict_answer, test_doc in zip(results, self.__test_data):
                if predict_answer == test_doc["answer"]:
                    correct += 1

            self.__save_model()

            return {"total_num": len(self.__test_data), "correct_num": correct}

        else:
            tokenize_msg = get_valid_tag(msg)

            feed_dict = {
                self.__original_sequence_in_char.name: np.zeros(dtype=np.int64, shape=(self.__setting["batch_size"], self.__setting["max_sentence_len"], self.__setting["max_word_len"])),
                self.__original_sequence_lengths.name: np.zeros(dtype=np.int64, shape=(self.__setting["batch_size"],))
            }
            feed_dict[self.__original_sequence_lengths.name][0] = len(tokenize_msg)
            for word_idx in range(len(tokenize_msg)):
                char_word = kor2eng(tokenize_msg[word_idx])
                for char_idx in range(len(char_word)):
                    if char_word[char_idx] in self.__share["idx2char"]:
                        feed_dict[self.__original_sequence_in_char.name][0][word_idx][char_idx] = self.__share["idx2char"].index(char_word[char_idx])
                    else:
                        feed_dict[self.__original_sequence_in_char.name][0][word_idx][char_idx] = self.__share["idx2char"].index("UNKNOWN")

            output = self.__session.run(self.__output, feed_dict=feed_dict)
            result = output.tolist()[0]
            result = [{"answer": label, "val": val} for label, val in zip(self.__share["idx2label"], result)]
            result.sort(key=lambda x: x["val"], reverse=True)

            return result
