from matplotlib import pyplot as plt
import argparse
import numpy as np

def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='record_dir', default='./records/active/', help='record directory')
    parser.add_argument('-p', dest='on_screen', action='store_true', help='show graph on screen')
    parser.add_argument('-s', dest='dont_save', action='store_true',
        help='reduce the dataset size for a short test of the code')
    return parser.parse_args()


if __name__ == '__main__':
    cl_args = parse_cl_args()

    metric_file = cl_args.record_dir + '/metric'
    try:
        file = open(metric_file, 'r')
    except FileNotFoundError:
        print('Metric file {} not found'.format(metric_file))
        exit()

    epochs = []
    t_loss_list = []
    v_loss_list = []
    cer_list = []
    wer_list = []
    cnt = 0
    
    line = file.readline()
    while line:
        cnt += 1
        line = line.strip()
        metrics = line.split(',')

        try:
            t_loss = float(metrics[0].strip())
            v_loss = float(metrics[1].strip())
            cer = float(metrics[2].strip())
            wer = float(metrics[3].strip())
        except:
            print('Invalid metric: {}'.format(metrics))
            exit()
        
        epochs.append(cnt)
        t_loss_list.append(t_loss)
        v_loss_list.append(v_loss)
        cer_list.append(cer)
        wer_list.append(wer)

        line = file.readline()
    # end of while
    file.close()

    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    t_handle, = plt.plot(epochs, t_loss_list)
    v_handle, = plt.plot(epochs, v_loss_list)
    plt.xticks(np.arange(0, cnt, 5.0))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epochs vs Loss')
    plt.grid(True)
    plt.legend([t_handle, v_handle], ['training', 'validation'], loc = 'upper right')

    plt.subplot(122)
    cer_handle, = plt.plot(epochs, cer_list)
    wer_handle, = plt.plot(epochs, wer_list)
    plt.xticks(np.arange(0, cnt, 5.0))
    plt.xlabel('Epochs')
    plt.ylabel('Error rate')
    plt.title('Epoch vs Error rate')
    plt.grid(True)
    plt.legend([cer_handle, wer_handle], ['CER', 'WER'], loc = 'upper right')

    plt.subplots_adjust(bottom=0.2)
    
    if not cl_args.dont_save:
        img = cl_args.record_dir + "/graph.png"
        plt.savefig(img) # save to file
    if cl_args.on_screen:
        plt.show()
