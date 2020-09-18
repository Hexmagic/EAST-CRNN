#!/usr/bin/python
# encoding: utf-8

import pdb
import torch

from config.crnn_cfg import directory


class LabelEncoder(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """
    def __init__(self, ignore_case=False):
        self._ignore_case = ignore_case
        self.words = directory
        self.words["-"] = len(self.words)

    def __call__(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text.
        """

        length = []
        result = []
        text = text.decode("utf-8", "strict")
        for item in text:
            length.append(len(item))
            r = []
            for char in item:
                if char in self.words:
                    index = self.words[char]
                    r.append(index)
                else:
                    r.append(0)
            result.append(r)

        max_len = 0
        for r in result:
            if len(r) > max_len:
                max_len = len(r)

        result_temp = []
        for r in result:
            for i in range(max_len - len(r)):
                r.append(0)
            result_temp.append(r)

        text = result_temp
        return (torch.LongTensor(text), torch.LongTensor(length))


class LabelDecoder(object):
    def __init__(self, ignore_case=False):
        self._ignore_case = ignore_case
        self.words = {}
        for k, v in directory.items():
            self.words[v] = k

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert (
                t.numel() == length
            ), "text with length: {} does not match declared length: {}".format(
                t.numel(), length)
            if raw:
                return "".join([self.words[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.words[t[i].item()])
                return "".join(char_list)
        else:
            # batch mode
            assert (
                t.numel() == length.sum()
            ), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(t[index:index + l],
                                torch.LongTensor([l]),
                                raw=raw))
                index += l
            return texts
