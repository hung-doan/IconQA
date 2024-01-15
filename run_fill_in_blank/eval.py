import os
import sys
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import utils
from dataset import Dictionary, IconQAFeatureDataset
import base_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str).replace('_ ', '')


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


def compute_test_acc(pred_logits, qIds, dataloader):
    acc = 0
    results = {}
    results_detail = {}
    N = len(dataloader.dataset)
    assert N == qIds.size(0)

    for i, data in enumerate(dataloader.dataset.entries):
        pid = data['question_id']
        gt_ans = data['answer']
        pred_ans = get_answer(pred_logits[i], dataloader)
        results[pid] = pred_ans
        is_correct = pred_ans == gt_ans
        results_detail[pid] = is_correct
        if is_correct:
            acc += 1
    acc = 100 * acc / N
    return acc, results, results_detail


@torch.no_grad()
def evaluate(model, dataloader, output):
    print("\nThe model is testing")
    utils.create_dir(output)
    model.train(False)

    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    pbar = tqdm(total=len(dataloader))

    for i, (v, q, a, qid) in enumerate(dataloader):
        pbar.update(1)
        batch_size = v.size(0)

        v = v.to(device)
        q = q.to(device)
        logits,v_att_weight = model(v, q)
        #print(v_att_weight)

        ques_ids = torch.IntTensor([int(ques_id) for ques_id in qid])
        pred[idx:idx + batch_size, :].copy_(logits.data)
        qIds[idx:idx + batch_size].copy_(ques_ids)
        idx += batch_size

        if args.debug:
            print("\nQuestion id:", qid[0])
            print(get_question(q.data[0], dataloader))
            print(get_answer(logits.data[0], dataloader))

    pbar.close()

    acc, results, results_detail = compute_test_acc(pred, qIds, dataloader)
    print("\nTest acc: %.3f" % acc)

    # save results to json file
    data = {}
    data['accuracy'] = acc
    data['args'] = {}
    for arg in vars(args):
        data['args'][arg] = getattr(args, arg)
    data['results'] = results
    data['results_detail'] = results_detail
    with open("{}/{}_{}.json".format(args.output, args.label, args.model), 'w') as f:
        json.dump(data, f, indent = 2, separators=(',', ': '))

    print("Done!")

def generate_attention_map(att_mat, batch_size, question_id):
     
    #att_mat = att_mat.squeeze(1)
    

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1).to(torch.device('cpu'))

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    
    print("\n att_mat:", att_mat.shape)
    print("\n residual_att:", att_mat.shape)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
        
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    print("\n joint_attentions v:", v.shape)
    
    
    # Normalize all into the smallest patch.
    patch_map = [1, 2, 3, 4, 7]
    v_skip = 0
    for pa in patch_map:
        v_grid_size = pa
        if v_grid_size != 1:
            visualize_attention_map(aug_att_mat, v, v_skip, v_grid_size, question_id)
        v_skip += v_grid_size * v_grid_size
    
    
    
def visualize_attention_map(aug_att_mat, v, v_skip, v_grid_size, question_id): 
    # question_id = 103
    #
    # Visualize
    #
    im = Image.open(f"{args.input}/iconqa_data/iconqa/test/fill_in_blank/{question_id}/image.png")
    question_data = json.load(open(f"{args.input}/iconqa_data/iconqa/test/fill_in_blank/{question_id}/data.json"))
    # image transformer
 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    im = transform(im)


    #grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    grid_size = v_grid_size; # Hard coded for now to get the last layer of [1, 2, 3, 4, 7]
    #mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    patch_skip = v_skip + 1
    patch_take_to =  patch_skip + grid_size*grid_size 
    
    print("\n grid_size:", grid_size)
    print("\n patch_skip:", patch_skip)
    print("\n patch_take_to:", patch_take_to)
    mask = v[0, patch_skip:patch_take_to].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    result = (mask * im).astype("uint8")

    # Plot figure
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    fig.suptitle(f'Question {question_id}: {question_data["question"]}')

    ax1.set_title('Original')
    ax2.set_title('Attention Map')
    _ = ax1.imshow(im)
    _ = ax2.imshow(result)
    #fig.show()
    
    fig.savefig(f'../attention_maps/fill_in_blank/test/{question_id}-{v_grid_size}.png')

hook_activations = {}
vit_attention_hook_name = 'vit_attention'
def get_activation(name):
    def hook(model, input, output):
        if name == vit_attention_hook_name:
            print('\n Executed hook: vit_attention')
            hook_activations[name].append(model.attn_weights)
        else:
            hook_activations[name] = output.detach()
    return hook

def hook_attention_weight(model):
    hook_activations[vit_attention_hook_name] = []
    model.v_trm_net.transformer.net[0].fn.fn.register_forward_hook(get_activation(vit_attention_hook_name))

def inspect_attention(batch_size, question_id):
    print("Inspecing the attention for question:", question_id)
    #print(model)
    print(len(hook_activations[vit_attention_hook_name]))
    
    # It dould be [datasize, headsize, ?,?] why 80x80 in the last dim
    # https://jacobgil.github.io/deeplearning/vision-transformer-explainability
    # is that mean 80 = number of path 79 + 1 for class token.
    print(hook_activations[vit_attention_hook_name][0].size())
    
    #print(model.v_att.dropout.state_dict())
    #att_mat = torch.stack(model.v_att.dropout).squeeze(1)
    #print(att_mat)
    att_mat=hook_activations[vit_attention_hook_name]
    # att_mat[0] = first images
    generate_attention_map(att_mat[0], batch_size, question_id)

def parse_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--task', type=str, default='fill_in_blank')
    # input and output
    parser.add_argument('--feat_label', type=str, default='resnet101_pool5_79_icon',
                        help = 'resnet101_pool5_79_icon: icon pretrained model')
    parser.add_argument('--input', type=str, default='../data')
    parser.add_argument('--model_input', type=str, default='../saved_models/fill_in_blank')
    parser.add_argument('--output', type=str, default='../results/fill_in_blank')
    # model and label
    parser.add_argument('--model', type=str, default='patch_transformer_ques_bert')
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--check_point', type=str, default='best_model.pth')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument("--debug", default=False, help='debug mode or not')
    # transformer
    parser.add_argument('--num_patches', type=int, default=79)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--patch_emb_dim', type=int, default=768)
    # language model
    parser.add_argument('--lang_model', type=str, default='bert-small',
                        choices=['bert-tiny', 'bert-mini', 'bert-small', 'bert-medium', 'bert-base', 'bert-base-uncased'])
    parser.add_argument('--max_length', type=int, default=34)

    # filter testing data
    parser.add_argument('--test_ids', type=str, default=[], nargs='*')

    # Inspect attention    
    parser.add_argument("--inspect-att", default=False, help='Show attention map or not')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    for arg in vars(args):
        print(arg, ':',  getattr(args, arg))

    torch.backends.cudnn.benchmark = True

    # dataset
    dictionary = Dictionary.load_from_file(args.input + '/dictionary.pkl') # load dictionary
    eval_dset = IconQAFeatureDataset('test', args.task, args.feat_label, args.input,
                                  dictionary, args.lang_model, args.max_length, args.test_ids) # generate test data
    batch_size = args.batch_size

    # data loader
    test_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=4)

    # build the model
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid, args.lang_model,
                                             args.num_heads, args.num_layers, 
                                             args.num_patches, args.patch_emb_dim)

    # load the trained model
    model_path = os.path.join(args.model_input, args.label, args.check_point)
    print('\nloading %s' % model_path)
    model_data = torch.load(model_path)
    model.load_state_dict(model_data.get('model_state', model_data), strict=False) # ignore missing key(s) in state_dict

    # GPU
    device = torch.device('cuda:' + args.gpu)
    model.to(device)

    # model testing
    hook_attention_weight(model)
    evaluate(model, test_loader, args.output)

    # Inspect the attention layer
    if args.inspect_att:
        inspect_attention(batch_size, args.test_ids[0])

    print("Done.", flush=True)