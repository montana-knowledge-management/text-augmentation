# Text augmentation

[![tests](https://github.com/montana-knowledge-management/text-augmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/text-augmentation/actions)
[![codecov](https://codecov.io/gh/montana-knowledge-management/text-augmentation/branch/main/graph/badge.svg?token=KMYGW7NHWH)](https://codecov.io/gh/montana-knowledge-management/text-augmentation)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/montana-knowledge-management/text-augmentation.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/montana-knowledge-management/text-augmentation/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/montana-knowledge-management/text-augmentation.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/montana-knowledge-management/text-augmentation/context:python)

## What is this repository good for?

Augmenting data is a common solution in response to real world scenarios where either not enough training data is
available or the dataset is highly imbalanced. This is the case when augmenting data can come to your rescue.

## Currently implemented augmentation techniques

* Easy Data Augmentation: For more information, please refer to the paper of Wei and
  Zou [here](https://arxiv.org/abs/1901.11196?ref=hackernoon.com).
    * Random deletion
    * Random swap
    * Random synonym insertion
    * Synonym replacement
* Word vector based replacement
    * `gensim` and `fastText`
    * several selection criteria, like: pick first, pick uniformly, pick with user-defined weighted probabilities, pick
      based on cosine distance weights.
* Contextual-embedding based augmentation (soon to come)

## Additional features

###Protected words 

One common problem would be in case of augmentation that there might be certain words that should not be modified by any
means. To tackle this problem, `text-augmentation` package offers a list in which you can define the words to remain
intact during the augmentation procedure. 
