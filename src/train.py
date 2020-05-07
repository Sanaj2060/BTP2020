import tensorflow as tf
import numpy as np
import argparse
import time
import os
import shutil
import pdb

import model
import data
import param
import metrics
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
train_details_path = 'records/active/'
checkpoint_path = 'records/active/checkpoints'

def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', dest='epochs', required=True, type=int, help='number of epochs')
    parser.add_argument('-l', dest='lang_code', required=True, help='language code')
    parser.add_argument('-i', dest='interval', default=5, type=int, help='interval to save checkpoints')
    parser.add_argument('-R', dest='restart', action='store_true',
        help='delete checkpoint, training_details and restart training')
    parser.add_argument('-V', dest='no_validate', action='store_true',
        help='do not perform validation during training')
    parser.add_argument('-r', dest='reverse', action='store_true',
        help='reverse the example')
    parser.add_argument('-S', dest='short_test', action='store_true',
        help='reduce the dataset size for a short test of the code')
    parser.add_argument('-D', dest='debug', action='store_true',
        help='debugging mode')
    return parser.parse_args()
cl_args = parse_cl_args()
    
if cl_args.restart:
    # prevent accidental restart
    opt = input('\n\nRestart training? y/n: ')
    if (opt == 'y') or (opt == 'Y'):
        opt = input('Confirm? y/n: ')
        if not ((opt == 'y') or (opt == 'Y')):
            print('Exiting')
            exit()
    else:
        print('Exiting')
        exit()
        
if cl_args.debug:
    pdb.set_trace()

if cl_args.epochs < 1:
    print('Invalid epochs: ', cl_args.epochs)
    exit()

# Get the datasets
dataset = data.Data(cl_args.lang_code, cl_args.reverse)
train_dataset, test_dataset, val_dataset = dataset.get_dataset(cl_args.short_test)

# Transformer network
transformer = model.Transformer(param.NUM_LAYERS, param.D_MODEL, param.NUM_HEADS, param.DFF,
    input_vocab_size = dataset.inp_vocab_size,
    target_vocab_size = dataset.tar_vocab_size, 
    pe_input = param.PAD_SIZE, 
    pe_target = param.PAD_SIZE,
    rate=param.DROPOUT
)

train_loss = tf.metrics.Mean(name='train_loss')
optimizer = utils.optimizer

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None), dtype=tf.float32)
]
@tf.function(input_signature = train_step_signature)
def train_step(inp, tar_inp, tar_real):
    enc_padding_mask, combined_mask, dec_padding_mask = utils.create_masks(inp, tar_inp)  
    # shape(inp) = (batch_size, pad_size)
    # shape(predictions) = (batch_size, pad_size, tar_vocab_size)
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, 
                                    True, 
                                    enc_padding_mask, 
                                    combined_mask, 
                                    dec_padding_mask)
        loss = metrics.loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)

def val_step(val_dataset):
    val_perf = metrics.PerformanceMetrics()
    
    val_dataset = utils.mk_eval_dataset(val_dataset)
    
    i = 0
    for inp, ref_real in val_dataset.items():
        i += 1
        # shape(pred) = (pad_size, tar_vocab_size)
        # shape(ref_real) = (ref_size, pad_size)
        pred = utils.predict(inp, transformer) 
        
        val_perf(ref_real, pred)

        if (i + 1) % (100) == 0:
            loss, cer, wer = val_perf.result()
            print ('\tValidation update\t Loss: {:.2f}\tWord acc: {:.2f}\tChar acc: {:.2f}'.format(loss, 1 - wer, 1 - cer))
    return val_perf.result()


if __name__ == '__main__':    
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=15)

    # Store training details
    dataset_size = dataset.get_dataset_size()

    tmp_lang_code = cl_args.lang_code
    if cl_args.reverse:
      tmp_lang_code += '-reverse'
    extra = 'lang_code: {}\tepochs: {},\ttrain_size: {},\tval_size: {}'.format(
        tmp_lang_code, cl_args.epochs, dataset_size[0], dataset_size[0])
    train_details = utils.TrainDetails(train_details_path, extra=extra)

    if cl_args.restart:
        print('\nRemoving train_details and checkpoints')
        train_details.rm_details_file()
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)

        print('\nCreating new train_details files')
        train_details.create_req_files()
    else:
        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('\nLatest checkpoint restored: ', ckpt_manager.latest_checkpoint)

    start = time.time()
    best_loss, best_checkpoint = utils.get_best()

    for epoch in range(cl_args.epochs):
        print('\nEPOCH: ', epoch+1)

        train_loss.reset_states()

        for batch, dataset in enumerate(train_dataset):
            inp, tar_inp, tar_real = dataset[:, 0, :]    , dataset[:, 1, :], dataset[:, 2, :]
            
            # Training
            train_step(inp, tar_inp, tar_real)
            
            if (batch + 1) % 100 == 0:
                print ('\tTraining batch update\tEpoch: {}\t Batch: {}\t Loss: {:.2f}\t'.format(epoch + 1, batch + 1,
                    train_loss.result()))                
        print ('\nEpoch: {}\ttrain_loss: {:.4f}\t'.format(epoch + 1, train_loss.result()))
            
        # Validation
        if not cl_args.no_validate:
            print('\nValidating...') 
            v_loss, cer, wer = val_step(val_dataset)
            print ('\nEpoch: {}\ttrain_loss: {:.4f}\tval_loss  : {:.4f}\t'.format(epoch + 1, train_loss.result(), v_loss))
        else:
            v_loss, cer, wer = -0.0, -0.0, -0.0        

        # save metrics
        train_details.save_metric('{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(train_loss.result(), v_loss, cer, wer))

        #  Saving checkpoint if the loss improves
        if not cl_args.no_validate:
            curr_loss = v_loss
        else:
            curr_loss = train_loss.result()
        # prefering the latest
        if curr_loss <= best_loss:
            ckpt_save_path = ckpt_manager.save()
            best_loss = curr_loss
            best_checkpoint = ckpt_manager.latest_checkpoint
            print ('\nSaving best checkpoint for epoch {} at {}\n'.format(epoch+1, ckpt_save_path))

        if (epoch + 1) % cl_args.interval == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('\nSaving interval checkpoint for epoch {} at {}\n'.format(epoch+1, ckpt_save_path))

            if not cl_args.no_validate:
                curr_loss = v_loss
            else:
                curr_loss = train_loss.result()
            # prefering the latest
            if curr_loss <= best_loss:
                best_loss = curr_loss
                best_checkpoint = ckpt_manager.latest_checkpoint
    # end of for loop
    
    # save checkppoint for last epoch
    ckpt_save_path = ckpt_manager.save()
    print ('\nSaving latest checkpoint for epoch {} at {}\n'.format(epoch+1, ckpt_save_path))
    if not cl_args.no_validate:
        curr_loss = v_loss
    else:
        curr_loss = train_loss.result()
    # prefering the latest
    if curr_loss <= best_loss:
        best_loss = curr_loss
        best_checkpoint = ckpt_manager.latest_checkpoint

    # Recording best checkpoint number
    utils.record_best(best_loss, best_checkpoint)

    epoch_time_taken = time.time() - start
    total_time_taken = train_details.save_elapsed_time(epoch_time_taken)
    print ('\nTime taken for this session: {}\n'.format(utils.get_time(epoch_time_taken)))
    print ('Total time taken: {}\n'.format(utils.get_time(total_time_taken)))
