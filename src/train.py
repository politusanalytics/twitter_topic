from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef
import torch
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from commons import MLP
from data import get_examples, TransformersData
from torch.utils.data import DataLoader
import numpy as np
import time
import random
from tqdm import tqdm
import json
import sys
# import pandas as pd

# INPUTS
pretrained_transformers_model = sys.argv[1] # For example: "xlm-roberta-base"
seed = int(sys.argv[2])
module_name = sys.argv[3]
device_ids = [int(i) for i in sys.argv[4].split(",")]

# MUST SET THESE VALUES
max_seq_length = 64
batch_size = 64
dev_ratio = 0.1

repo_path = "/home/username/twitter_topic"
train_filename = "{}/data/adjudicated_20230222/{}/train.json".format(repo_path, module_name)
test_filename = "{}/data/adjudicated_20230222/{}/test.json".format(repo_path, module_name)

only_test = False # Only perform testing
predict = False # Predict instead of testing
has_token_type_ids = False

if module_name == "welfare":
    label_list = ["social_policy", "labour_and_employment", "education",
                  "health_and_public_health", "disability", "housing"]
elif module_name == "democracy":
    label_list = ["elections_and_voting", "justice_system", "human_rights",
                  "regime_and_constitution", "kurdish_question"]
elif module_name == "big5":
    label_list = ["internal_affairs", "national_defense", "corruption", "foreign_affairs", "economy"]
elif module_name == "municipal":
    train_filename = f"{repo_path}/data/20230703_municipal/train.json"
    test_filename = f"{repo_path}/data/20230703_municipal/test.json"
    label_list = ["urban_public_infrastructure", "social_and_welfare_services",
                  "environment_and_public_health", "housing", "animal_welfare",
                  "local_politics", "culture"]
else:
    raise("Module name {} is not known!".format(module_name))

label_to_idx = {}
idx_to_label = {}
for (i, label) in enumerate(label_list):
    label_to_idx[label] = i
    idx_to_label[i] = label

# SETTINGS
learning_rate = 2e-5
dev_metric = "f1_macro"
num_epochs = 30
dev_set_splitting = "random" # random, or any filename
use_gpu = True
# device_ids = [0, 1, 2, 3, 4, 5, 6, 7] # if not multi-gpu then pass a single number such as [0]
positive_threshold = 0.5 # Outputted probabilities bigger than this number is considered positive in case of binary classifications
return_probabilities = False # whether or not to return probabilities instead of labels when predicting
model_path = "{}_{}_{}.pt".format(pretrained_transformers_model.replace("/", "_"), module_name, seed)

# optional, used in testing
classifier_path = ""# repo_path + "/models/best_models/20220528_classifier_sentence-transformers_paraphrase-xlm-r-multilingual-v1_44.pt"
encoder_path = ""#repo_path + "/models/best_models/20220528_encoder_sentence-transformers_paraphrase-xlm-r-multilingual-v1_44.pt"

if not classifier_path:
    classifier_path =  repo_path + "/models/classifier_" + model_path
if not encoder_path:
    encoder_path =  repo_path + "/models/encoder_" + model_path

if return_probabilities:
    from scipy.special import softmax

if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:%d"%(device_ids[0]))
else:
    device = torch.device("cpu")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device.type == "cuda":
    torch.cuda.manual_seed_all(seed)

tokenizer = None
criterion = torch.nn.BCEWithLogitsLoss()

def test_model(encoder, classifier, dataloader):
    all_preds = []
    all_label_ids = []
    eval_loss = 0
    nb_eval_steps = 0
    for val_step, batch in enumerate(dataloader):
        if has_token_type_ids:
            input_ids, input_mask, token_type_ids, label_ids = tuple(t.to(device) for t in batch)
            embeddings = encoder(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[1]
        else:
            input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
            embeddings = encoder(input_ids, attention_mask=input_mask)[1]

        with torch.no_grad():
            out = classifier(embeddings)
            tmp_eval_loss = criterion(out, label_ids)

        eval_loss += tmp_eval_loss.mean().item()

        # curr_preds = torch.sigmoid(out).detach().cpu().numpy().tolist()
        # curr_preds = [[int(x >= positive_threshold) for x in preds] for preds in curr_preds]
        # all_preds += curr_preds

        # label_ids = label_ids.to('cpu').numpy().tolist()
        # all_label_ids += label_ids

        # Multi-label f1 does not work in this version. So we just flatten our matrix.
        curr_preds = torch.sigmoid(out).detach().cpu().numpy().flatten().tolist()
        curr_preds = [int(x >= positive_threshold) for x in curr_preds]
        all_preds += curr_preds

        label_ids = label_ids.to('cpu').numpy().flatten().tolist()
        all_label_ids += label_ids

        nb_eval_steps += 1

    precision, recall, f1, _ = precision_recall_fscore_support(all_label_ids, all_preds, average="macro") #, labels=list(range(0,len(label_list))))
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(all_label_ids, all_preds, average="micro") #, labels=list(range(0,len(label_list))))
    mcc = matthews_corrcoef(all_preds, all_label_ids)
    eval_loss /= nb_eval_steps
    result = {"precision_macro": precision,
              "recall_macro": recall,
              "f1_macro": f1,
              "precision_micro": precision_micro,
              "recall_micro": recall_micro,
              "f1_micro": f1_micro,
              "mcc": mcc}

    return result, eval_loss

# Test for labels individually
def test_model2(encoder, classifier, dataloader):
    all_preds = {i: [] for i in range(len(label_list))}
    all_label_ids = {i: [] for i in range(len(label_list))}
    eval_loss = 0
    nb_eval_steps = 0
    for val_step, batch in enumerate(dataloader):
        if has_token_type_ids:
            input_ids, input_mask, token_type_ids, label_ids = tuple(t.to(device) for t in batch)
            embeddings = encoder(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[1]
        else:
            input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
            embeddings = encoder(input_ids, attention_mask=input_mask)[1]

        with torch.no_grad():
            out = classifier(embeddings)
            tmp_eval_loss = criterion(out, label_ids)

        eval_loss += tmp_eval_loss.mean().item()

        label_ids = label_ids.to('cpu').numpy().tolist()
        curr_preds = torch.sigmoid(out).detach().cpu().numpy().tolist()
        for i, preds in enumerate(curr_preds):
            for ii, pred in enumerate(preds):
                all_preds[ii].append(int(pred >= positive_threshold))
                all_label_ids[ii].append(int(label_ids[i][ii]))

        nb_eval_steps += 1

    result = {}
    for i in all_preds.keys():
        curr_label = idx_to_label[i]

        curr_label_preds = all_preds[i]
        curr_label_gold = all_label_ids[i]

        precision, recall, f1, _ = precision_recall_fscore_support(curr_label_gold, curr_label_preds, average="macro")
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(curr_label_gold, curr_label_preds, average="micro")
        mcc = matthews_corrcoef(curr_label_preds, curr_label_gold)
        result["{}_precision_macro".format(curr_label)] = precision
        result["{}_recall_macro".format(curr_label)] = recall
        result["{}_f1_macro".format(curr_label)] = f1
        result["{}_precision_micro".format(curr_label)] = precision_micro
        result["{}_recall_micro".format(curr_label)] = recall_micro
        result["{}_f1_micro".format(curr_label)] = f1_micro
        result["{}_mcc".format(curr_label)] = mcc

    return result, eval_loss


def model_predict(encoder, classifier, dataloader):
    all_preds = []
    for val_step, batch in enumerate(dataloader):
        if has_token_type_ids:
            input_ids, input_mask, token_type_ids = tuple(t.to(device) for t in batch)
            embeddings = encoder(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[1]
        else:
            input_ids, input_mask = tuple(t.to(device) for t in batch)
            embeddings = encoder(input_ids, attention_mask=input_mask)[1]

        with torch.no_grad():
            out = classifier(embeddings)

        curr_preds = torch.sigmoid(out).detach().cpu().numpy().tolist()
        if return_probabilities:
            curr_preds = [[round(float(x), 4) for x in preds] for preds in curr_preds]
        else:
            curr_preds = [[idx_to_label[i] for i, x in enumerate(preds) if x >= positive_threshold] for preds in curr_preds]
        all_preds += curr_preds

    return all_preds


def build_model(train_examples, dev_examples, pretrained_model, n_epochs=10, curr_model_path="temp.pt"):
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    encoder = AutoModel.from_pretrained(pretrained_model)
    classifier = MLP(encoder.config.hidden_size, encoder.config.hidden_size*4, len(label_list))

    train_dataset = TransformersData(train_examples, label_to_idx, tokenizer, max_seq_length=max_seq_length, has_token_type_ids=has_token_type_ids)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dev_dataset = TransformersData(dev_examples, label_to_idx, tokenizer, max_seq_length=max_seq_length, has_token_type_ids=has_token_type_ids)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=batch_size)


    classifier.to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda" and len(device_ids) > 1:
        encoder = torch.nn.DataParallel(encoder, device_ids=device_ids)
    encoder.to(device)

    optimizer = torch.optim.AdamW(list(classifier.parameters()) + list(encoder.parameters()), lr=learning_rate)
    num_train_steps = int(len(train_examples) / batch_size * num_epochs)
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps = 0,
    #                                             num_training_steps = num_train_steps)

    best_score = -1e6
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = 0
        encoder.train()
        classifier.train()

        print("Starting Epoch %d"%(epoch+1))
        global_step = 0
        train_loss = 0.0
        nb_tr_steps = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            if has_token_type_ids:
                input_ids, input_mask, token_type_ids, label_ids = tuple(t.to(device) for t in batch)
                embeddings = encoder(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[1]
            else:
                input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
                embeddings = encoder(input_ids, attention_mask=input_mask)[1]

            out = classifier(embeddings)
            loss = criterion(out, label_ids)

            loss.backward()
            global_step += 1
            nb_tr_steps += 1

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            # scheduler.step()
            encoder.zero_grad()
            classifier.zero_grad()

            train_loss += loss.item()

        train_loss /= nb_tr_steps
        elapsed = time.time() - start_time

        # Validation
        encoder.eval()
        classifier.eval()
        result, val_loss = test_model(encoder, classifier, dev_dataloader)
        result["train_loss"] = train_loss
        result["dev_loss"] = val_loss
        result["elapsed"] = elapsed
        print("***** Epoch " + str(epoch + 1) + " *****")
        for key in sorted(result.keys()):
            print("  %s = %.6f" %(key, result[key]))

        print("Val score: %.6f" %result[dev_metric])

        if result[dev_metric] > best_score:
            print("Saving model!")
            torch.save(classifier.state_dict(), repo_path + "/models/classifier_" + curr_model_path)
            encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder  # To handle multi gpu
            torch.save(encoder_to_save.state_dict(), repo_path + "/models/encoder_" + curr_model_path)
            best_score = result[dev_metric]

        print("------------------------------------------------------------------------")

    return encoder, classifier

if __name__ == '__main__':
    if dev_set_splitting == "random":
        train_examples = get_examples(train_filename)
        random.shuffle(train_examples)
        dev_split = int(len(train_examples) * dev_ratio)
        dev_examples = train_examples[:dev_split]
        train_examples = train_examples[dev_split:]
    else: # it's a custom filename
        train_examples = get_examples(train_filename)
        random.shuffle(train_examples)
        dev_examples = get_examples(dev_set_splitting)

    if not only_test:
        encoder, classifier = build_model(train_examples, dev_examples, pretrained_transformers_model, n_epochs=num_epochs, curr_model_path=model_path)
        classifier.load_state_dict(torch.load(repo_path + "/models/classifier_" + model_path))
        if torch.cuda.device_count() > 1 and device.type == "cuda" and len(device_ids) > 1:
            encoder.module.load_state_dict(torch.load(repo_path + "/models/encoder_" + model_path))
        else:
            encoder.load_state_dict(torch.load(repo_path + "/models/encoder_" + model_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_transformers_model)
        encoder = AutoModel.from_pretrained(pretrained_transformers_model)
        classifier = MLP(encoder.config.hidden_size, encoder.config.hidden_size*4, len(label_list))

        classifier.to(device)
        encoder.to(device)
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))

        if torch.cuda.device_count() > 1 and device.type == "cuda" and len(device_ids) > 1:
            encoder = torch.nn.DataParallel(encoder, device_ids=device_ids)

    encoder.eval()
    classifier.eval()

    if predict:
        test_examples = get_examples(test_filename, with_label=False)
        test_dataset = TransformersData(test_examples, label_to_idx, tokenizer, max_seq_length=max_seq_length, has_token_type_ids=has_token_type_ids, with_label=False)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

        all_preds = model_predict(encoder, classifier, test_dataloader)
        with open(test_filename, "r", encoding="utf-8") as f:
            test = [json.loads(line) for line in f.read().splitlines()]

        with open(repo_path + "/out.json", "w", encoding="utf-8") as g:
            for i, t in enumerate(test):
                t["prediction"] = all_preds[i]
                g.write(json.dumps(t) + "\n")

    else:
        test_examples = get_examples(test_filename)
        test_dataset = TransformersData(test_examples, label_to_idx, tokenizer, max_seq_length=max_seq_length, has_token_type_ids=has_token_type_ids)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

        result, test_loss = test_model2(encoder, classifier, test_dataloader)
        result["test_loss"] = test_loss

        print("***** TEST RESULTS *****")
        for key in sorted(result.keys()):
            print("  %s = %.6f" %(key, result[key]))

        result, test_loss = test_model(encoder, classifier, test_dataloader)
        for key in sorted(result.keys()):
            print("  %s = %.6f" %(key, result[key]))
        print("TEST SCORE: %.6f" %result[dev_metric])
