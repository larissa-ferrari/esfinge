import json
import random

def loader_tf(dataset_file=None):
  dataset_tmp = {int(d['id']): d for d in json.load(dataset_file)}
  rev_dataset = {}
  black_list = set()
 
  for key in sorted(dataset_tmp.keys()):
    value = dataset_tmp[key]
    rev_dataset.setdefault(value['text'], set()).add(key)
    if len(rev_dataset[value['text']]) > 1:
      black_list.add(int(value['id']))
  del rev_dataset

  dataset = []
  for d in dataset_tmp.values():
    if int(d['id']) not in black_list:
      dataset.append(d)
  del dataset_tmp
  del black_list

  random.shuffle(dataset)

  return dataset
