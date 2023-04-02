'''
in alignment.txt files, 
Each line's format: utterance id, followed by the ground truth words and finally the end time for each word. E.g.:
84-121123-0000 ",GO,,DO,YOU,HEAR," "0.490,0.890,1.270,1.380,1.490,1.890,2.09" 
84-121123-0001 ",BUT,IN,LESS,THAN,FIVE,MINUTES,THE,STAIRCASE,GROANED,BENEATH,AN,EXTRAORDINARY,WEIGHT," "0.270,0.420,0.530,0.730,0.870,1.100,1.460,1.580,2.080,2.490,2.780,2.860,3.470,3.830,3.99" 

Note: separated by spaces,
keep everything in the same structure, BUT
reverse sentence entirely,
comma separate time steps but reverse ordering
'''
from typing import List
def reverse_alignment_txt(path: str) -> None:
  with open(path, 'r') as file:
    # read a list of lines into data
    data = file.readlines()  # each element is a txt line
  
  # modify data
  reversed_data = []
  for i, line in enumerate(data):
    id_words_endTimes = line.split()

    new_line = [""] * 3
    new_line[0] = id_words_endTimes[0]
    new_line[1] = id_words_endTimes[1][::-1]

    # remove beginning and ending "
    endTimes_str = id_words_endTimes[2].replace("\"", "")

    endTimes: List[str] = endTimes_str.split(",")
    endTimes.reverse()

    # add beginning and ending "
    if i == len(data) - 1:
      new_line[2] = "\"" + ','.join(endTimes) + "\""
    else:
      new_line[2] = "\"" + ','.join(endTimes) + "\"\n"
    
    reversed_data.append(" ".join(new_line))

  
  # and write everything back
  with open(path, 'w') as file:
    file.writelines( reversed_data )


import os, fnmatch
for root, dirnames, filenames in os.walk("/fs/scratch/PAS2400/TTS_baseline/training_data/LibriSpeech/train-clean-100/"):
    for filename in filenames:
        if filename.endswith("alignment.txt"):
            # print(os.path.join(root, filename))
            reverse_alignment_txt(os.path.join(root, filename))
