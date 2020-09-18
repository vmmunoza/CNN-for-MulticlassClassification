import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

df_train = pd.read_pickle('df_train.pkl.gzip', compression='gzip')    
color_list=['red','black','green']
def plot_trace(trace):
    dt = df_train.loc[df_train['trace_id']==trace]
    row_item = dt.loc[dt['trace_id'] == trace,['target','label','trace_id']].to_string().split('\n')[1].split()
    fig, ax = plt.subplots(3, 1, figsize = (20,4), sharex=True)
    for enum_tt, tt in enumerate(['E','N','Z']):
        signal = dt[tt][int(trace.replace('trace_',''))]
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        ax[enum_tt].plot(signal,'k', color=color_list[enum_tt])
        ax[enum_tt].set_ylabel('Amplitude {}'.format(tt))
    plt.suptitle('Target / Label / Trace_id   :   {} / {} / {}'.format(row_item[1],row_item[2],row_item[3]))    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.xlim([0,4000])
    plt.show();

    for col, val in zip(list(dt.columns),dt.values[0]):
        if col not in ['E','N','Z']:
            print('{} : {}'.format(col, val))

def plot_CNN_learning(cnn_net):
    fig,ax=plt.subplots(ncols=2,nrows=1, figsize=(20,5))
    epoch_number = len(cnn_net.history['loss'])
    for key in cnn_net.history.keys():
        if (key != 'lr') & (key[:3]!='val') :
            ax[0].plot(range(epoch_number), cnn_net.history[key], linewidth=2.0,label=key)
        elif (key != 'lr') & (key[:3]=='val') :
            ax[1].plot(range(epoch_number), cnn_net.history[key],linewidth=2.0, label=key)    
    ax[0].legend(fontsize=14)
    ax[0].set_xlabel('Epochs', fontsize=15)
    ax[0].set_yscale('log')
    ax[0].set_title('Training', fontsize=18)

    #ax[1].tick_params(axis="x", labelsize=24)    
    ax[0].tick_params(axis='both', which='major', labelsize=13)
    ax[0].tick_params(axis='both', which='minor', labelsize=13)
    ax[1].tick_params(axis='both', which='major', labelsize=13)
    ax[1].tick_params(axis='both', which='minor', labelsize=13)
    
    ax[1].legend(fontsize=14)
    ax[1].set_xlabel('Epochs', fontsize=15)
    ax[1].set_yscale('log')
    ax[1].set_title('Validation', fontsize=18)
    
    ax[1].set_xlim([0,20])
    ax[0].set_xlim([0,20])
    ax[1].set_ylim([1e-1,1e1])
    ax[0].set_ylim([1e-1,1e1])
    plt.show();
    
def plot_confustion_matrix(y_test, y_predicted, df, prob=False):
    if prob:
        cms = confusion_matrix(y_test.argmax(1), y_predicted.argmax(1))
    else:
        cms = confusion_matrix(y_test, y_predicted)
    test_score = np.trace(cms) / np.sum(cms)
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111)
    im = ax.imshow(np.transpose(cms), interpolation="nearest", cmap="PuRd")
    rows = cms.shape[0]
    cols = cms.shape[1]
    for x in range(0, rows):
        for y in range(0, cols):
            value = int(cms[x, y])
            ax.text(x, y, value, color="black", ha="center", va="center",fontsize=13)
    plt.title("Real v/s Predicted Data", fontsize=17)
    plt.colorbar(im)
    
    df_labels = df[['target','label']].drop_duplicates().sort_values(by='target')
    classes_values = [tt for tt in df_labels['target']]
    classes_labels = [tt for tt in df_labels['label']]

    plt.xticks(classes_values, classes_labels, fontsize=13)
    plt.yticks(classes_values, classes_labels, fontsize=13)
    plt.xlabel("Real data",fontsize=15)
    plt.ylabel("Predicted data",fontsize=15)
    b, t = plt.ylim()
    b += 0.0 # Add 0.5 to the bottom
    t -= 0.0 # Subtract 0.5 from the top
    plt.ylim(b, t) # update ylim(bottom, top) 
    plt.show();              