import numpy as np
from tensorflow.keras.layers import AveragePooling2D


nelx = 5
nely = 5
chunkSize = 2

x = np.array([  [1,1,0,1,0],
                [1,1,1,1,1],
                [0,1,1,1,0],
                [1,1,1,1,1],
                [0,1,1,1,1]],dtype='float32')

pool2D = AveragePooling2D(pool_size=(chunkSize,chunkSize),strides=(chunkSize,chunkSize),padding='same')

activityMap = pool2D(np.reshape(x,(1,nelx,nely,1)).astype('float32'))
newShape = activityMap.shape[1]*activityMap.shape[2]

activeValues = np.reshape(np.asarray(activityMap),(newShape))
print("\n")
print(activeValues)

activeValues = np.where(activeValues == 1,0,activeValues)

numNonZeros = len(activeValues[activeValues != 0])
numZeros = len(activeValues) - len(activeValues[activeValues != 0])
print(activeValues)
print(numZeros)

indexes = np.argsort(activeValues)[numZeros:]

print(indexes)

binaryChoice = np.zeros((len(indexes)),dtype='float32')
n = len(binaryChoice)
for i in range(1,n+1):
        binaryChoice[n-i] = 1/(2**(i))
binaryChoice[0] += 1/(2**(n))

print(binaryChoice)
print()
pick = np.random.choice(indexes,p=binaryChoice)
print(pick)
pick = np.random.choice(indexes,p=binaryChoice)
print(pick)
pick = np.random.choice(indexes,p=binaryChoice)
print(pick)
pick = np.random.choice(indexes,p=binaryChoice)
print(pick)
pick = np.random.choice(indexes,p=binaryChoice)
print(pick)
pick = np.random.choice(indexes,p=binaryChoice)
print(pick)

print("done")