# -*- coding: utf-8 -*-
# Natural Language Toolkit: Interface to the TreeTagger POS-tagger
#
# Copyright (C) Mirko Otto
# Author: Mirko Otto <dropsy@gmail.com>

"""
A Python module for interfacing with the Treetagger by Helmut Schmid.
"""

import os
from subprocess import Popen, PIPE

from nltk.internals import find_binary, find_file
from nltk.tag.api import TaggerI
from sys import platform as _platform

_treetagger_url = 'http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/'

_treetagger_languages = ['bulgarian', 'dutch', 'english', 'estonian', 'finnish', 'french', 'galician', 'german',
                         'italian', 'polish', 'russian', 'slovak', 'slovak2', 'spanish', 'portuguese']

# Acceptable parts of speech tags by language: only nouns, abbreviations and unknown tags
polish = "subst xxx ign brev burk adj".split(" ")
dutch = "noun adj".split(" ")
french = "abr nom nam adj".split(" ")
italian = "abr fw nom npr adj".split(" ")
english = "nn fw np jj".split(" ")

class TreeTagger(TaggerI):
    r"""
    A class for pos tagging with TreeTagger. The default encoding used by TreeTagger is utf-8. The input is the paths to:
     - a language trained on training data
     - (optionally) the path to the TreeTagger binary

    This class communicates with the TreeTagger binary via pipes.
    """

    def __init__(self, path_to_home=None, language='german',
                 verbose=False, abbreviation_list=None):
        """
        Initialize the TreeTagger.

        :param path_to_home: The TreeTagger binary.
        :param language: Default language is german.

        The encoding used by the model. Unicode tokens
        passed to the tag() and batch_tag() methods are converted to
        this charset when they are sent to TreeTagger.
        The default is utf-8.

        This parameter is ignored for str tokens, which are sent as-is.
        The caller must ensure that tokens are encoded in the right charset.
        """
        treetagger_paths = ['.', '/usr/bin', '/usr/local/bin', '/opt/local/bin',
                            '/Applications/bin', '~/bin', '~/Applications/bin',
                            '~/work/tmp/treetagger/cmd', '~/TreeTagger/cmd']
        treetagger_paths = list(map(os.path.expanduser, treetagger_paths))
        self._abbr_list = abbreviation_list
        self.language = language

        if language in _treetagger_languages:
            if _platform == "win32":
                treetagger_bin_name = 'tag-' + language
            else:
                treetagger_bin_name = 'tree-tagger-' + language
        else:
            raise LookupError('Language not in language list!')

        try:
            self._treetagger_bin = find_binary(
                treetagger_bin_name, path_to_home,
                env_vars=('TREETAGGER', 'TREETAGGER_HOME'),
                searchpath=treetagger_paths,
                url=_treetagger_url,
                verbose=verbose)
        except LookupError:
            print('NLTK was unable to find the TreeTagger bin!')

    def tag(self, sentences):
        """Tags a single sentence: a list of words.
        The tokens should not contain any newline characters.
        """

        # Write the actual sentences to the temporary input file
        if isinstance(sentences, list):
            _input = '\n'.join((x for x in sentences))
        else:
            _input = sentences

        # Run the tagger and get the output
        if (self._abbr_list is None):
            p = Popen([self._treetagger_bin],
                      shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        elif (self._abbr_list is not None):
            p = Popen([self._treetagger_bin, "-a", self._abbr_list],
                      shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE)

        # (stdout, stderr) = p.communicate(bytes(_input, 'UTF-8'))
        (stdout, stderr) = p.communicate(str(_input).encode('utf-8'))

        # Check the return code.
        if p.returncode != 0:
            print(stderr)
            raise OSError('TreeTagger command failed!')

        treetagger_output = stdout.decode('UTF-8')

        # Output the tagged sentences
        tagged_sentences = []
        for tagged_word in treetagger_output.strip().split('\n'):
            tagged_word_split = tagged_word.split('\t')
            tagged_sentences.append(tagged_word_split)

        return tagged_sentences

    def is_acceptable_portuguese_tag(self, tag):
        return (tag.startswith('n') and not tag.endswith('g0')) or tag.startswith('a')

    def is_acceptable_russian_tag(self, tag):
        return tag.startswith('n') or tag.startswith('a')

    def is_acceptable_german_tag(self, tag):
        return tag.startswith('n') or tag.startswith('adj')

    def is_acceptable_tag(self, tag, acceptable_tags):
        for acceptable_tag in acceptable_tags:
            if acceptable_tag in tag:
                return True
        return False

    def is_acceptable_word(self, word):
        try:
            word_tag = self.tag(word)[0][1].lower()
            return {
                'english': self.is_acceptable_tag(word_tag, english),
                'french': self.is_acceptable_tag(word_tag, french),
                'german': self.is_acceptable_german_tag(word_tag),
                'portuguese': self.is_acceptable_portuguese_tag(word_tag),
                'italian': self.is_acceptable_tag(word_tag, italian),
                'polish': self.is_acceptable_tag(word_tag, polish),
                'russian': self.is_acceptable_russian_tag(word_tag),
                'dutch': self.is_acceptable_tag(word_tag, dutch)
            }[self.language]
        except:
            return True
