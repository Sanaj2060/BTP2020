import model
import data
import utils
import param
import metrics

import time
import argparse
import tensorflow as tf


def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', dest='lang_code', required=True, help='language code')
    parser.add_argument('-r', dest='reverse', action='store_true',
        help='reverse the example')
    parser.add_argument('-S', dest='short_test', action='store_true',
        help='reduce the dataset size for a short test of the code')
    return parser.parse_args()

def eval_step(transformer, test_dataset, eval_file):
    eval_perf = metrics.PerformanceMetrics()

    test_dataset = utils.mk_eval_dataset(test_dataset)
    
    i = 0
    for inp, ref_real in test_dataset.items():
        i += 1
        # shape(pred) = (pad_size, tar_vocab_size)
        # shape(ref_real) = (ref_size, pad_size)
        pred = utils.predict(inp, transformer) 
        
        eval_perf(ref_real, pred)

        pred = tf.argmax(pred, axis = -1)
        tr_inp  = dataset.tokenizer.inp_decode(inp) #.numpy()) 
        tr_pred = dataset.tokenizer.tar_decode(pred.numpy())
        for real in ref_real:
            tr_real = dataset.tokenizer.tar_decode(real.numpy())
            eval_file.write('{}, {}, {}, {}\n'.format(tr_inp, tr_real, tr_pred, str(tr_real == tr_pred)))

        if (i + 1) % (100) == 0:
            loss, cer, wer = eval_perf.result()
            print ('\tEvaluation update\t Loss: {:.2f}\t CER: {:.2f}\t WER: {:.2f}'.format(
                loss, cer, wer))
    eval_loss, cer, wer = eval_perf.result()
    eval_file.write('test_dataset size: {}\n'.format(i))
    eval_file.write('Loss: {:.4f}\tCER: {:.4f}\tWER: {:.4f}'.format(eval_loss, cer, wer))
    return eval_loss, cer, wer


if __name__ == '__main__':
    cl_args = parse_cl_args()

    train_details_path = 'records/active/'
    checkpoint_path = 'records/active/checkpoints'
    best_checkpoint_path = 'records/active/checkpoints/best'
    
    # Get the datasets
    dataset = data.Data(cl_args.lang_code, cl_args.reverse)
    _, test_dataset, _ = dataset.get_dataset(cl_args.short_test)

    # Transformer network
    transformer_network = model.Transformer(param.NUM_LAYERS, param.D_MODEL, param.NUM_HEADS, param.DFF,
        input_vocab_size = dataset.inp_vocab_size,
        target_vocab_size = dataset.tar_vocab_size, 
        pe_input = param.PAD_SIZE, 
        pe_target = param.PAD_SIZE,
        rate=param.DROPOUT
    )

    # Restoring from best checkpoint for evaluating
    ckpt = tf.train.Checkpoint(transformer=transformer_network, optimizer=utils.optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=15)
    try:
        with open(best_checkpoint_path, 'r') as file:
            best_checkpoint = file.read().split(' ')[1]
    except:
        print('Cannot open best checkpoint file: ', best_checkpoint_path)
        exit()

    ckpt.restore(best_checkpoint)
    print ('\nBest checkpoint restored: ', best_checkpoint)

    try:
        eval_file = open('./records/active/eval', 'w')
    except:
        print('Cannot open eval file: /records/active/eval')
        exit()

    start = time.time()
    print('\nEvaluating ...\n')
    eval_loss, cer, wer = eval_step(transformer_network, test_dataset, eval_file)
    print('\nAfter evaluation:\tLoss: {:.4f}\tCER: {:.4f}\t WER: {:.4f}'.format(eval_loss, cer, wer))
    eval_time_taken = time.time() - start
    print ('\nTime taken for evaluation: {}\n'.format(utils.get_time(eval_time_taken)))