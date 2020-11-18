import os
import torch
from src import logger, trainer, data, models
from torch.optim.sgd import SGD
from flair.embeddings import WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
import flair
import argparse
import datetime

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, default=os.path.join("trained_models", 
                                                                      datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
                                                                     ))
    parser.add_argument("--model", type=str)
    parser.add_argument("--device", choices=('cuda:0', 'cuda:1', 'cpu'), default='cuda:0')

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--word-dropout", type=float, default=0.05)
    parser.add_argument("--locked-dropout", type=float, default=0.5)
    parser.add_argument("--relearn-embeddings", action='store_true', default=False)
    parser.add_argument("--use-crf", action="store_true", default=False)
    parser.add_argument("--cell-type", type=str, choices=("LSTM", "RNN", "GRU"), default="LSTM")
    parser.add_argument("--bidirectional", action="store_true", default=False)

    parser.add_argument("--embeddings", type=str, choices=("glove", "glove+pooled"), default="glove")

    parser.add_argument("--epochs", type=int, default=150)

    parser.add_argument("--train-with-dev", action='store_true', default=False)
    parser.add_argument("--embeddings-storage-mode", choices=('cpu', 'gpu', 'none'), default='cpu')

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-shuffle", action="store_true", default=False)

    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--learning-rate-min", type=float, default=1e-4)
    parser.add_argument("--learning-rate-decay", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--step-size", type=int, default=0)
    
    parser.add_argument("--no-verbose", action='store_true', default=False)
    parser.add_argument("--monitor-train", action='store_true', default=False)
    parser.add_argument("--monitor-dev", action='store_true', default=False)
    parser.add_argument("--monitor-test", action='store_true', default=False)
    parser.add_argument("--print-every-batch", type=int, default=50)

    parser.add_argument("--save-every-epoch", type=int, default=0)
    parser.add_argument("--checkpoint-name", type=str, default=None)

    # for mixup:
    parser.add_argument("--n-passes", type=int, default=2)
    parser.add_argument("--sort", action='store_true', default=False)
    parser.add_argument("--lambdas-generator", type=str, choices=('beta',), default="beta")
    parser.add_argument("--rho", type=float, default=0)
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--beta", type=float, default=2)

    parser.add_argument("--data-path", type=str, default="data/conll_03")
    parser.add_argument("--tag-name", type=str, default="ner")
    parser.add_argument("--tag-scheme", type=str, default="iobes")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = process_args()

    if not os.path.exists(args.base_path):
        os.makedirs(args.base_path)

    with open(os.path.join(args.base_path, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

    logger.log_dir = args.base_path + '/log.txt'
    logger.verbose = not args.no_verbose
    
    assert(args.step_size == 0 or args.patience == 0)
    
    device = torch.device(args.device)
    flair.device = device

    if args.tag_name == "ner":
        corpus = data.Conll_Corpus(path=args.data_path, tag_name=args.tag_name, tag_scheme=args.tag_scheme)

    corpus.build()
    tag_dic = corpus.create_tags_dictionary()

    if args.embeddings == "glove":
        embedding_types = [
            WordEmbeddings('glove'),
        ]
    elif args.embeddings == "glove+pooled":
        embedding_types = [
            WordEmbeddings('glove'),
            PooledFlairEmbeddings('news-forward', pooling='min'),
            PooledFlairEmbeddings('news-backward', pooling='min'),
        ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    if args.lambdas_generator == 'beta':
        lambdas_generator_params = {'alpha': args.alpha, 
                                    'beta' : args.beta,
                                    'rho'  : args.rho,
                                   }

    model = getattr(models, args.model)(embeddings, args.hidden_size, tag_dic,
                                        device=device,
                                        use_crf=args.use_crf,
                                        relearn_embeddings=args.relearn_embeddings,
                                        word_dropout=args.word_dropout,
                                        locked_dropout=args.locked_dropout,
                                        sort=args.sort,
                                        lambdas_generator=args.lambdas_generator,
                                        lambdas_generator_params=lambdas_generator_params,
                                        tag_name=args.tag_name,
                                        cell_type=args.cell_type,
                                        bidirectional=args.bidirectional,)

    model_trainer = trainer.trainer(model, corpus, SGD, args.base_path)

    history = model_trainer.train(args.epochs,
                                  n_passes=args.n_passes,
                                  train_with_dev=args.train_with_dev,
                                  mixup_training=not(args.model == "Normal"),
                                  embedding_storage_mode=args.embeddings_storage_mode,
                                  batch_size=args.batch_size,
                                  shuffle=not args.no_shuffle,
                                  learning_rate=args.learning_rate,
                                  learning_rate_decay=args.learning_rate_decay,
                                  learning_rate_min=args.learning_rate_min,
                                  patience=args.patience,
                                  step_size=args.step_size,
                                  monitor_train=args.monitor_train,
                                  monitor_dev=args.monitor_dev,
                                  monitor_test=args.monitor_test,
                                  print_every_batch=args.print_every_batch,
                                  save_every_epoch=args.save_every_epoch,
                                  checkpoint_name=args.checkpoint_name,
                                 )

