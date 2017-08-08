# Sentence Splitter Development


This repository is a bit of a mess but I figured it was mainly for exploratory code and wouldn't be used by anyone else. This readme is meant to point out the most relevant files:

###TensorflowStyle.py
This file contains the code to train the k-nearest-neighbor algorithm and record the parameter optimization results in a specified output file. The algorithm is trained using the notes found in `/ClinicalNotes`.

###TestBuiltinSplitter.py
This file is a script to evaluate the pyConText.helpers built-in sentence splitter using the training notes found in ClinicalNotes.

###/ClinicalNotes
This directory contains the test notes. Sentence boundaries have been marked using the `^` character. This character was chosen because it was not present in any of the corpus notes.

###Other Files
The other files are mainly support files for TensorflowStyle.py or TestBuiltinSplitter.py or temporary files that I haven't taken the time to delete.

