import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from classes.modeling_bert import BertForTokenClassification
from classes.configration_bert import BertConfig
from classes.tokenization_bert import BertTokenizer
from classes import file_utils
from runs.utils_ner import convert_examples_to_features, get_labels, read_examples_from_file
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import logging
import random
from tqdm import tqdm, trange
from classes.optimization import AdamW,get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from seqeval.metrics import precision_score, recall_score, f1_score

SEED = 1234
logger = logging.getLogger(__name__)
logging.basicConfig(filename='logfile/logger.log', level=logging.DEBUG)

data_dir = './data/ner'
model_type = 'bert'
model_name_or_path = ''
output_dir = './output/ner'
labels = [
        "O", 
        "B-DATE", 
        "I-DATE",  
        "B-PERSON", 
        "I-PERSON", 
        "B-ARTIFACT", 
        "I-ARTIFACT", 
        "B-LOCATION", 
        "I-LOCATION",
        "B-NUMBER",
        "I-NUMBER",
        "B-PERCENT",
        "I-PERCENT",
        "B-TIME",
        "I-TIME",
        "B-EVENT",
        "I-EVENT",
        "B-ORGANIZATION",
        "I-ORGANIZATION",
        "B-OTHER",
        "I-OTHER",
        "B-MONEY",
        "I-MONEY",
        ]
config_name = ""
tokenizer_name = ""
cache_dir = ""
max_seq_length = 128
per_gpu_train_batch_size = 8
per_gpu_eval_batch_size = 8
gradient_accumulation_steps = 1
learning_rate = 5e-5
weight_decay = 0.0
adam_epsilon = 1e-8
max_grad_norm = 1.0
num_train_epochs = 10.0
max_steps = -1
warmup_steps = 0
logging_steps = 50
save_steps = 50
evaluate_during_training = True
local_rank = -1

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train(train_dataset, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    tb_writer = SummaryWriter()
    train_batch_size = 8
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    
    t_total = len(train_dataloader) // 1 * 3

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
    '''
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    '''

    # multi-gpu training (should be after apex fp16 initialization)
    '''
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    '''
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", 3)
    logger.info("  Instantaneous batch size per GPU = %d", 8)
    logger.info("  Gradient Accumulation steps = %d", 1)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(num_train_epochs), desc="Epoch", disable=False)
    set_seed(SEED)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to('cpu') for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}
            if model_type != "distilbert":
                inputs["token_type_ids"] = batch[2] if "bert" in ["bert", "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            '''
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            '''
            '''
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
            '''
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % 1 == 0:
                '''
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                '''
                    
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if 50 > 0 and global_step % logging_steps == 0:
                    # Log metrics
                    if local_rank == -1 and evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(model, tokenizer, labels, pad_token_label_id, mode="dev")
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / logging_steps, global_step)
                    logging_loss = tr_loss

                if local_rank in [-1, 0] and save_steps > 0 and global_step % save_steps == 0:
                    # Save model checkpoint
                    out_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(out_dir)
                    #torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", out_dir)

            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break
        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break
    tb_writer.close()
    return global_step, tr_loss / global_step


def evaluate(model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples('./data/ner',tokenizer, labels, pad_token_label_id, mode=mode)

    eval_batch_size = per_gpu_eval_batch_size * max(1, 1)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # multi-gpu evaluate
    '''
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    '''

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}
            if model_type != "distilbert":
                inputs["token_type_ids"] = batch[2] if model_type in ["bert", "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            #if args.n_gpu > 1:
            #    tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list)
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list

def load_and_cache_examples(data_dir, tokenizer, labels, pad_token_label_id, mode):

    #logger.info("Creating features from dataset file at %s", data_dir)
    examples = read_examples_from_file(data_dir, mode)
    features = convert_examples_to_features(examples, labels, 512, tokenizer,
                                            cls_token_at_end=bool('bert' in ["xlnet"]),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if 'bert' in ["xlnet"] else 0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool('bert' in ["roberta"]),
                                            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                            pad_on_left=bool('bert' in ["xlnet"]),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if 'bert' in ["xlnet"] else 0,
                                            pad_token_label_id=pad_token_label_id
                                            )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def main():
    
    set_seed(SEED)
    
    num_labels = len(labels)
    pad_token_label_id = CrossEntropyLoss().ignore_index
    config_class = BertConfig(vocab_size_or_config_json_file=file_utils.get_bert_congfig_path())
    model_class = BertForTokenClassification(config_class)
    config = config_class.from_pretrained(pretrained_model_name_or_path='./pre_train_model/bert/pytorch/',num_labels=num_labels,cache_dir="")
    model = model_class.from_pretrained(pretrained_model_name_or_path='./pre_train_model/bert/pytorch/',
                                        from_tf=False,
                                        config=config,
                                        cache_dir="")
    tokenizer_class = BertTokenizer('./pre_train_model/bert/pytorch/vocab.txt')
    tokenizer = tokenizer_class.from_pretrained('./pre_train_model/bert/pytorch/',
                                                do_lower_case=True,
                                                cache_dir="")
    model.to(device)

    train_dataset = load_and_cache_examples('./data/ner', tokenizer, labels, pad_token_label_id, mode="train")
    global_step, tr_loss = train(train_dataset, model, tokenizer, labels, pad_token_label_id)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    
    
    # Create output directory if needed
    if not os.path.exists(output_dir) and local_rank in [-1, 0]:
        os.makedirs(output_dir)

    logger.info("Saving model checkpoint to %s", output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    #torch.save(args, os.path.join(output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    
    tokenizer = tokenizer_class.from_pretrained(output_dir, do_lower_case=True)
    checkpoints = [output_dir]
    #if args.eval_all_checkpoints:
        #checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
        #logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        
        model = model_class.from_pretrained(checkpoint,config=config)
        model.to(device)
        result, _ = evaluate(model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step)
        if global_step:
            result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
        results.update(result)
    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        for key in sorted(results.keys()):
            writer.write("{} = {}\n".format(key, str(results[key])))


    tokenizer = tokenizer_class.from_pretrained('./pre_train_model/bert/pytorch/',
                                                do_lower_case=True,
                                                cache_dir="")
    model = model_class.from_pretrained(output_dir,config=config)
    model.to(device)
    result, predictions = evaluate(model, tokenizer, labels, pad_token_label_id, mode="test")
    # Save results
    output_test_results_file = os.path.join(output_dir, "test_results.txt")
    with open(output_test_results_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("{} = {}\n".format(key, str(result[key])))
    # Save predictions
    output_test_predictions_file = os.path.join(output_dir, "test_predictions.txt")
    with open(output_test_predictions_file, "w") as writer:
        with open(os.path.join(data_dir, "test.txt"), "r") as f:
            example_id = 0
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    writer.write(line)
                    if not predictions[example_id]:
                        example_id += 1
                elif predictions[example_id]:
                    output_line = line.split()[0] + " " + predictions[example_id].pop(0) + "\n"
                    writer.write(output_line)
                else:
                    logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])

    return results
        
if __name__ == "__main__":
    main()