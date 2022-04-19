"""
Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --no-char-decoder                       do not use the character decoder
"""
import math
import sys
import numpy as np
import torch
import torch.nn.utils
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from typing import List, Tuple, Dict, Set, Union

from nmt_model import Hypothesis, NMT
from utils import read_corpus, batch_iter


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] in ('<s>', '<SOS>'):
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references], [hyp.value for hyp in hypotheses])
    return bleu_score


@torch.no_grad()
def evaluate_ppl(model, dev_data, batch_size=32):
    """ Evaluate on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns validation loss
    """
    was_training = model.training
    model.eval()

    cum_loss = 0

    for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
        loss = -model(src_sents, tgt_sents).sum()

        cum_loss += loss.item()

    if was_training:
        model.train()

    return cum_loss / len(dev_data)


def train(args: Dict):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    """
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])

    clip_grad = float(args['--clip-grad'])

    vocab = args['--vocab']

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab, no_char_decoder=args['--no-char-decoder'])
    model.train()

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init))
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    hist_train_losses = []
    hist_val_losses = []
    
    print('begin Maximum Likelihood training')
    for epoch in range(args['--max-epoch']):
        
        cum_train_loss = 0
        best_val_bleu = 0
        
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):

            optimizer.zero_grad()

            example_losses = - model(src_sents, tgt_sents)
            batch_loss = example_losses.sum()
            loss = batch_loss / len(src_sents)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            
            cum_train_loss += batch_loss
        
        hist_train_losses.append(cum_train_loss/len(train_data))
        
        # perform validation
        val_loss = evaluate(model, dev_data, batch_size=128)
        hist_val_losses.append(val_loss)
    
    return hist_train_losses, hits_val_losses


def decode(model: NMT, args: Dict[str, str]):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    print("load test source sentences from [{}]".format(args['--test-src']))
    test_data_src = read_corpus(args['--test-src'], source='src')
    if args['--test-tgt']:
        print("load test target sentences from [{}]".format(args['--test-tgt']))
        test_data_tgt = read_corpus(args['--test-tgt'], source='tgt')

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['--test-tgt']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print('Corpus BLEU: {}'.format(bleu_score * 100))

    with open('output.txt', 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')

            
def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training()
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding'):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
            hypotheses.append(example_hyps)

    if was_training:
        model.train()

    return hypotheses
