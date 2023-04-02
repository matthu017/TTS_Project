from typing import List
def reverse_trans_txt(path: str) -> None:
  with open(path, 'r') as file:
    # read a list of lines into data
    data = file.readlines()  # each element is a txt line
  
  # modify data
  reversed_data = []
  for i, line in enumerate(data):
    id_words_list: List[str] = line.split()

    id = id_words_list[0]

    words_list = id_words_list[1:]
    words = " ".join(words_list)

    reversed_words = words[::-1]
    reversed_data.append(id + " " + reversed_words + "\n")
  
  # and write everything back
  rpath = path.replace("train-clean-100", "train-clean-100_reversed")
  rpath = path.replace("TTS_baseline", "TTS_baseline_cpy")
  with open(rpath, 'w') as file:
    file.writelines( reversed_data )

import os, fnmatch
for root, dirnames, filenames in os.walk("/fs/scratch/PAS2400/TTS_baseline/training_data/LibriSpeech/train-clean-100/"):
    for filename in filenames:
        if filename.endswith("trans.txt"):
            # print(os.path.join(root, filename))
            isExist = os.path.exists(root.replace("train-clean-100", "train-clean-100_reversed"))
            if not isExist:
                os.makedirs(root.replace("train-clean-100", "train-clean-100_reversed"))
            reverse_trans_txt(os.path.join(root, filename))