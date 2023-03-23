import numpy as np
import json
import matplotlib.pyplot as plt



def plotHistory_lite(hist):
    fig,ax = plt.subplots(1,1)
    ax.set_title("")
    keys = hist.keys()
    for key in keys:
        if('output' in key):
            minVal = min(hist[key])
            meanVal = np.mean(hist[key])
            maxVal = max(hist[key])
            if(minVal != maxVal):
                print("{}:\n\tmin:{}\n\tmean:{}\n\tmax:{}".format(key,minVal,meanVal,maxVal))
                #ax.plot(y,hist[key],linewidth=0.5,label=key)
        else:
            y = np.arange(1,len(hist[key])+1)
            normal = 1
            if(key == 'loss'):
                normal = 5
            ax.plot(y,np.array(hist[key])/normal,label=key)
            minVal = min(hist[key])
            meanVal = np.mean(hist[key])
            maxVal = max(hist[key])
            print("{}:\n\tmin:{}\n\tmean:{}\n\tmax:{}".format(key,minVal,meanVal,maxVal))
    
    plt.legend()#bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.show()

def main():
    name = 'trainHistory_'

    numHistoryFolder = 0

    #hist1 = json.load(open(name + str(0),'r'))
    hist1 = json.load(open("Model_m9_340_epoch_history.hist",'r'))

    print(hist1.keys())
    for i in range(0,numHistoryFolder):
        newHist = json.load(open(name + str(i),'r'))
        for key in hist1.keys():
            for val in newHist[key]:
                hist1[key].append(val)
    
    #print(hist1)
    plotHistory_lite(hist1)
    #plotHistory_lite(json.load(open("Model_m9_200_epoch_trainin_run_history.hist",'r')))

    #json.dump(hist1,open("Model_m9_340_epoch_history.hist",'w'))




if(__name__ == "__main__"):
    main()