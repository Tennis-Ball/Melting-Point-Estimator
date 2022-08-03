def reduce(dataX, dataY):
    minY = min(dataY)
    dataY = dataY - minY  # set minimum to 0

    THRESHOLD = 290
    bucketSize = 10
    buckets = [0] * int(max(dataY) // bucketSize) + [0]
    newX = []
    newY = []

    for i, mp in enumerate(dataY):
        if buckets[int(mp / bucketSize)] <= THRESHOLD:
            newX.append(dataX[i])
            newY.append(dataY[i] + minY)
            buckets[int(mp / bucketSize)] += 1
    
    print(buckets)
    return newX, newY
