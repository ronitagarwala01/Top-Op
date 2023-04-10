import numpy as np
import pandas as pd
import matplotlib.pyplot as plt










def main():
    df = pd.read_csv("DataFull.csv")
    columns = ["nelx","nely","did converge","Youngs modulus",
            "circle 1 x","circle 1 y","circle 1 radius","force 1 x","force 1 y",
            "circle 2 x","circle 2 y","circle 2 radius","force 1 x.1","force 2 y",
            "circle 3 x","circle 3 y","circle 3 radius","force 3 x","force 3 y",
            "True Mass","True Compliance","True Stress",
            "Predicted Mass","Predicted Stress","Predicted Compliance"]

    columnsToPCA = columns[:20]

    matrix = df[columnsToPCA].to_numpy(dtype='float32')
    matrix = matrix.T


    u,s,v_t = np.linalg.svd(matrix)

    error2D = np.sum(s[3:])
    error3D = np.sum(s[4:])

    print("PCA error for 2D representation: ",error2D)
    print("PCA error for 3D representation: ",error3D)

    pca2D = u[:,:3].T

    print(pca2D.shape)

    pcaCoords = np.zeros((3,len(df.index)))
    forceCoords = np.zeros((3,len(df.index)))
    distCoords = np.zeros((3,len(df.index)))
    parameterCoords = np.zeros((3,len(df.index)))


    status = []
    legend = []
    for i in range(len(df.index)):
        #print(df.iloc[i])
        vec = df.iloc[i][columnsToPCA].to_numpy(dtype='float32')
        vec = np.reshape(vec,(len(vec),1))
        #print(vec.shape)

        twodimOutput = np.reshape(pca2D @ vec,3)
        #print(twodimOutput)
        pcaCoords[0,i] = twodimOutput[0]
        pcaCoords[1,i] = twodimOutput[1]
        pcaCoords[2,i] = twodimOutput[2]


        compliance = df.iloc[i]['True Compliance']
        stress = df.iloc[i]['True Stress']
        predStress = df.iloc[i]['Predicted Compliance']#stress and compliance got swaped in saving. correcting them here
        predCompliance = df.iloc[i]['Predicted Stress']


        c_constraint = compliance >= predCompliance
        s_constraint = stress >= predStress


        #magnitude of force applied to each circle
        forceCoords[0,i] = np.sqrt((df.iloc[i]['force 1 x'] ** 2) + (df.iloc[i]['force 1 y'] ** 2))
        forceCoords[1,i] = np.sqrt((df.iloc[i]['force 1 x.1'] ** 2) + (df.iloc[i]['force 2 y'] ** 2))
        forceCoords[2,i] = np.sqrt((df.iloc[i]['force 3 x'] ** 2) + (df.iloc[i]['force 3 y'] ** 2))

        #Distance from each circle to the next
        distCoords[0,i] = np.sqrt((df.iloc[i]['circle 1 x'] - df.iloc[i]['circle 2 x'])** 2 + (df.iloc[i]['circle 1 y']- df.iloc[i]['circle 2 y'])** 2)
        distCoords[1,i] = np.sqrt((df.iloc[i]['circle 2 x'] - df.iloc[i]['circle 3 x'])** 2 + (df.iloc[i]['circle 2 y']- df.iloc[i]['circle 3 y'])** 2)
        distCoords[2,i] = np.sqrt((df.iloc[i]['circle 3 x'] - df.iloc[i]['circle 1 x'])** 2 + (df.iloc[i]['circle 3 y']- df.iloc[i]['circle 1 y'])** 2)

        
        #YOungs s_max, c_max
        parameterCoords[0,i] = compliance
        parameterCoords[1,i] = stress
        parameterCoords[2,i] = df.iloc[i]['Youngs modulus']

        #print("Stress: {} >= {} = {}".format(stress,predStress,s_constraint))
        #print("Comp.: {} >= {} = {}\n".format(compliance,predCompliance,c_constraint))

        if(c_constraint and s_constraint):
            legend.append('Comp. and Stres')
            status.append(0)
        elif(c_constraint):
            legend.append('Comp.')
            status.append(1)
        elif(s_constraint):
            legend.append('Stress')
            status.append(2)
        else:
            legend.append('None')
            status.append(3)


    colorVals = np.array(['red','green','blue','black'])
    markerVals = np.array(['1','o','^','s'])

    c = colorVals[np.array(status)]
    m = markerVals[status]
    legend = np.array(['Comp. and Stres','Comp.','Stress','None'])
    print(legend)
    print(colorVals)


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pcaCoords[0,:],pcaCoords[1,:],pcaCoords[2,:],c=colorVals[status])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(forceCoords[0,:],forceCoords[1,:],forceCoords[2,:],c=colorVals[status])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(parameterCoords[0,:],parameterCoords[1,:],parameterCoords[2,:],c=colorVals[status])
    plt.show()

if(__name__ == "__main__"):
    main()