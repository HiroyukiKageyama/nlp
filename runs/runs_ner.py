import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from classes.modeling_bert import BertForTokenClassification
from classes.configration_bert import BertConfig
from classes import file_utils
import random


SEED = 1234


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def main():
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    set_seed(SEED)
    labels = ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    num_labels = len(labels)
    pad_token_label_id = CrossEntropyLoss().ignore_index
    config_class = BertConfig(vocab_size_or_config_json_file=file_utils.get_bert_congfig_path())
    model_class = BertForTokenClassification(config_class)
    config = config_class.from_pretrained(pretrained_model_name_or_path='./pre_train_model/bert/pytorch/',num_labels=num_labels,cache_dir="")
    model = model_class.from_pretrained(pretrained_model_name_or_path='./pre_train_model/bert/pytorch/',
                                        from_tf=False,
                                        config=config,
                                        cache_dir="")
    print()
    
if __name__ == "__main__":
    main()