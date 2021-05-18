import configargparse
import shutil
import torch
import numpy as np
import os
import math
import data

from torch import optim
from torch.nn import functional as F

from etm import ETM
from utils import nearest_neighbors, get_topic_coherence, get_topic_diversity

def save_topics(m, vocab, num_topics, model_dir):
    m.eval()
    with torch.no_grad():
        f = open(os.path.join(model_dir, 'topics.txt'), 'w')
    gammas = m.get_beta()
    for k in range(num_topics):
        gamma = gammas[k]
        top_words = list(gamma.detach().cpu().numpy().argsort()[-50:][::-1])
        topic_words = [vocab[a] for a in top_words]
        f.write(' '.join(topic_words))
        f.write('\n')
    f.close()

if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description='The Embedded Topic Model')
    parser.add_argument('--model_dir', type=str, default='./results', help='to load model checkpoint and save topics')
    parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
    parser.add_argument('--data_path', type=str, default=None, help='directory containing data')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, _, _, _ = data.get_data(os.path.join(args.data_path))
    l = os.listdir(args.model_dir)
    for x in l:
        if x[:4]=='etm_':
            ckpt = os.path.join(args.model_dir, x)
            break
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)

    save_topics(model, vocab, args.num_topics, args.model_dir)

