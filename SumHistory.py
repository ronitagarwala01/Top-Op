import numpy as np
import json
import matplotlib.pyplot as plt

def softenArray(x,filterSize:int = 5):
    n = len(x)
    xNew = []
    count = 1 + (filterSize*2)
    for i in range(n):
        v = 0
        for j in range(i - filterSize, i+filterSize + 1):
            index = min(n-1,max(0,j))
            v += x[index]
        v /= count
        xNew.append(v)
    return xNew

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
            if('val' in key):
                yStart = 0
                while(hist[key][yStart] == 0):
                    yStart += 1
                print(yStart)
                y = np.arange(yStart,len(hist[key]))
                print(len(y))
                print(len(hist[key][yStart:]))
                x = hist[key][yStart:]
                ax.plot(y,np.array(softenArray(x,0)),label=key)
                a,b = np.polyfit(y,x,1)
                #print(p)
                #a = p[0]
                #b = p[1]
                x2 = np.array([0,len(hist[key])])
                y2 = a*(x2) + b
                #print(x2,y2)
                #ax.plot(x2,y2,label="{:.2e}".format(a))
                print("y = ({})x + ({})".format(a,b))
            else:
                y = np.arange(1,len(hist[key])+1)
                normal = 1
                if(key == 'loss'):
                    normal = 5
                ax.plot(y,np.array(softenArray(hist[key],0))/normal,label=key)
            minVal = min(hist[key])
            meanVal = np.mean(hist[key])
            maxVal = max(hist[key])
            print("{}:\n\tmin:{}\n\tmean:{}\n\tmax:{}".format(key,minVal,meanVal,maxVal))


    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.legend()#bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.savefig("loss.png")
    plt.show()


def main():
    name = 'trainHistory_'

    numHistoryFolder = 0

    #hist1 = json.load(open(name + str(0),'r'))
    hist1 = json.load(open("Model_m9_NewTrainingBatch_epoch110.hist",'r'))

    print(hist1.keys())
    for i in range(0,numHistoryFolder):
        try:
            newHist = json.load(open(name + str(i),'r'))
        except FileNotFoundError:
            print("File {} not found.".format(name + str(i)))
        for key in hist1.keys():
            for val in newHist[key]:
                hist1[key].append(val)
    
    #print(hist1)
    plotHistory_lite(hist1)
    #plotHistory_lite(json.load(open("Model_m9_200_epoch_trainin_run_history.hist",'r')))

    #json.dump(hist1,open("Model_m9_NewTrainingBatch_epoch_START_175_FIN_245.hist",'w'))

def mergeHist():
    hist1 = json.load(open("Model_m9_NewTrainingBatch_epoch110.hist",'r'))
    h2 = json.load(open("Model_m9_NewTrainingBatch_epoch_START_175_FIN_245.hist",'r'))


    #newHist = json.load(open(name + str(i),'r'))
    for key in hist1.keys():
        for val in h2[key]:
            hist1[key].append(val)

    plotHistory_lite(hist1)

def dropKeys():
    hist1 = json.load(open("Model_m9_440_epoch_history.hist",'r'))
    h2 = json.load(open("Model_m9_NewTrainingBatch_epoch110.hist",'r'))

    keysList = []
    for key in hist1.keys():
        keysList.append(str(key))
    
    for key in keysList:
        if('val' in key):
            for i in range(len(hist1[key])):
                hist1[key][i] = 0
            for val in h2[key]:
                hist1[key].append(val)
        else:    
            for val in h2[key]:
                hist1[key].append(val)
    
    plotHistory_lite(hist1)

if(__name__ == "__main__"):
    dropKeys()
    #mergeHist()
    #main()