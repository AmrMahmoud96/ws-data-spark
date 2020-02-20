from math import radians, cos, sin, asin, sqrt, pi, exp
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf, mean as _mean, stddev as _stddev, col, desc,collect_set,count
from collections import defaultdict
import sys

# for spark submit
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
sc.setLogLevel("ERROR")
spark = SparkSession(sc)


def sigmoid(x):
    """Applies sigmoid function for a given value x."""
    return 1 / (1 + exp(-x))

def readF(filepath):
    """Reads the file at the specified filepath.

    Reads the file at the specified filepath and returns the lines
    in the file as an array.

    Args:
        filepath (str) - the relative filepath of the file from the current working directory

    Returns:
        res - array of lines read from the file
        OR
        None - file does not exist in filepath
    """
    # init result array
    res=[]
    try:
        # try to open the file in the path provided in read mode
        f = open(filepath, 'r')
        # read lines into result array
        res = f.readlines()
        # close file
        f.close()
        # return results
        return res
    except IOError:
        # error occured, file not found, return none
        print('file not found')
        return None

def calcDist(oLat,oLong,dLat,dLong):
    """Calculates the Haversine Distance between two latlng points.

    Takes in origin (latitude, longitude) and destination (latitude,longitude)
    to return the haversine distance between the two points. Assumes
    spherical plane.

    Args:
        oLat (double) - the latitude of the origin point in degrees
        oLong (double) - the longitude of the origin point in degrees
        dLat (double) - the latitude of the destination point in degrees
        dLong (double) - the longitude of the destination point in degrees

    Returns:
        the absolute haversine distance between the origin and destination points
        rounded to 2 decimal places in KM
    """
    # turn latitude and longitude values to radians
    oLat,oLong,dLat,dLong = map(radians,[oLat,oLong,dLat,dLong])
    # calc difference in longitude points
    diffLong = dLong-oLong
    # calc difference in latitude points
    diffLat = dLat-oLat
    # find area of
    area = sin(diffLat/2)**2 + cos(oLat) * cos(dLat) * sin(diffLong/2)**2
    # find central angle
    angle = 2*asin(sqrt(area))
    # multiply by radius of earth (6371 KM), round to 2 decimals, return result
    return abs(round(angle*6371,2))

def findPath(originalStartPoints,end,completed,dependencies,topStart,allTasks):
    """Finds the path between start and end points given dependencies, starting
    positions, map of start points to parent start tasks, and set of all tasks.

    DP path finding function.

    Args:
        originalStartPoints (set) - set of starting points from input file
        end (str/int) - end point/task from input file
        completed (set) - set of completed tasks prior to starting (includes start points)
        dependencies (dict<str/int:list>) - dictionary that maps tasks to their dependent tasks
        topStart (dict) - dictionary that maps tasks before the starting tasks to their
            parent/highest level starting points
        allTasks (set) - a set of all tasks

    Returns:
        path (list) - a list, starting from the end point, that lists the tasks
            required to reach the end in REVERSE order.
    """
    # init empty path
    path = []
    # init visited set for quick lookup on path included nodes
    pathSet = set()
    # init set of empty required starting points
    start = set()
    # init queue with just the end point
    q = [end]
    while q:
        # while there are tasks queued, pop the first task into the curr var
        curr = q.pop(0)

        # if the current task is not in allTasks, continue
        if curr not in allTasks:
            continue

        # if the current task is completed prior to starting:
        if curr in completed:

            # check if current task is one of the input starting point and that
            # the parent/top starting point of this task is not already accounted for
            if curr in originalStartPoints and topStart[curr] not in start:
                # if above conditions are met, add the current task to the starting set
                start.add(curr)

            # otherwise, since the task is not a direct task or is already accounted for,
            # add the parent task to the starting set
            else:
                start.add(topStart[curr])

            # continue to next in queue
            continue
        # if the current node is already in the path, continue
        if curr in pathSet:
            continue

        # append the current task to the path and pathSet
        path.append(curr)
        pathSet.add(curr)

        # find the tasks that the current node depends on and add them to the queue
        q.extend(dependencies[curr])

    # return the path with a list of starting points at the end (reverse order)
    return path + list(start)



if __name__ == "__main__":

    # read csv to DataFrame objects
    df = spark.read.format('csv').options(header='true', inferSchema='true').load('/tmp/data/DataSample.csv')
    poi = spark.read.format('csv').options(header='true', inferSchema='true').load('/tmp/data/POIList.csv')

    #rename POI lat/long columns to avoid confusion in SQL join
    poi = poi.withColumnRenamed(" Latitude","pLat").withColumnRenamed("Longitude","pLong")

    ###### CLEANUP #####
    # drop duplicate timestamps and geoinfo
    df=df.dropDuplicates([' TimeSt', 'Latitude','Longitude'])
    poi=poi.dropDuplicates(['pLat','pLong'])

    print('1. CLEANUP')
    df.show(20)
    poi.show()

    #### LABEL #####
    # combine poi data with sample data by crossJoining poi with df
    fulldf = df.crossJoin(poi)

    # define function to calculate haversine distance between two geocordinates
    udfMinDist = udf(calcDist,DoubleType())

    # create new column as the haversine distance between poi and datapoint
    ndf=fulldf.withColumn("dist", udfMinDist(fulldf.Latitude, fulldf.Longitude,fulldf.pLat,fulldf.pLong))

    # groupby id and find the min distance (closest point), rejoin with combined df to get full data
    mindf = ndf.groupBy('_ID').min('dist').withColumnRenamed("min(dist)","dist").join(ndf,['_ID','dist'])
    # drop duplicates (two pois assigned to same request with equal distance)
    mindf = mindf.dropDuplicates(['_ID','dist'])

    print('2. LABEL')
    mindf.show(20)

    ##### ANALYSIS #####

    ### 1)
    # groupBy poiid and perform the mean and stddev aggregate functions on the distance column
    df_stats = mindf.groupBy('POIID').agg(
        _mean(col('dist')).alias('mean'),
        _stddev(col('dist')).alias('std')
    )
    # display results
    print('3.1 ANALYSIS (mean.stddev)')
    df_stats.show()


    ### 2)
    # get the furthest request for each POI and set that as radius of the circle to include all other requests
    # note that this would mean that all points would be on the same 2d plane (may not be the case in real life)
    maxDistRequests = mindf.groupBy('POIID').max('dist').withColumnRenamed('max(dist)','radius')
    # get # of requests for each POIID
    countRequests = mindf.groupBy('POIID').count()

    # join counts and radius
    pInfo = maxDistRequests.join(countRequests,['POIID'])

    # create new column 'density' with density = # of requests / area of circle
    pInfo = pInfo.withColumn('density',col('count') / (pi * col('radius')* col('radius')))

    print('3.2 ANALYSIS (density)')
    pInfo.show()

    ###### Data Engineering Tasks ########
    ### 4a)

    # set total number of requests
    totalRequests = mindf.count()

    # assume a standard normal distribution on the requests and apply sigmoid to the Z score
    for row in df_stats.collect():
        id,mean,std = row.POIID,row.mean,row.std
        count = int(countRequests.filter(countRequests.POIID == id).collect()[0][0])
        print(count)
        # extreme case -> no requests, set to -10
        if count == 0:
            popularity[id] = - 10
        # extreme case 2 -> all requests, set to 10
        if count == totalRequests:
            popularity[id] = 10
        # calculate z score
        Z = ( count - mean) / std
        # take sigmoid and map [0,1] to [-10,10]
        popularity[id] = (sigmoid(Z) * 20) - 10


    print('4a) popularities: ',popularity)

    ### 4b)

    # read inputs from the three files.
    inputs = readF('/tmp/data/question.txt')
    relations = readF('/tmp/data/relations.txt')
    tasks = readF('/tmp/data/task_ids.txt')

    # if inputs and relations worked, note there is also an IOException try/catch in the readF function
    if not (len(inputs)==2 and relations):
        print('error in reading files, unable to proceed with 4b. Ensure that all txt files are in data folder')
        sys.exit()

    # take out \n and take substring from index of : + 2 (to include space) to the end of the line, split by comma and turn to list of start points
    start = set(inputs[0].replace('\n','')[inputs[0].index(':')+2:].split(','))

    # take out \n and take substring from index of : + 2 (to include space) to the end of the line, should only be 1 end point
    end =  inputs[1].replace('\n','')[inputs[1].index(':')+2:]

    # make allTasks a set for quick lookup
    allTasks = set(tasks[0].replace('\n','').split(','))

    # initialize a set defaultdict
    dependencies = defaultdict(set)

    # for each relation
    for r in relations:
        # set pre as the string before the arrow and set the post to be the string after the arrow
        pre, post = r.split('->')[0],r.split('->')[1].replace('\n','')
        # add the dependent (pre) under the set of dependencies for the task (post)
        dependencies[post].add(pre)

    # completed will hold the list of tasks completed by the starting points
    completed = set()
    # top start will contain a map of each dependent task before the start to their top/furthest parent start
    topStart = {}
    while start:
        # for each starting point initialize a queue
        i = start.pop()
        queue = [i]
        while queue:
            # pop queue into current holder
            curr = queue.pop()
            # add current to the set of completed tasks
            completed.add(curr)
            # set the parent of the current task as the top level starting point
            topStart[curr] = i
            # if the current task is in start, remove it (so we don't double recurse)
            # note, this is more of a problem for topstart than it is for the completed set
            if curr in start:
                start.remove(curr)
            # append any dependent tasks to the queue and continue loop
            queue.extend(list(dependencies.get(curr,set())))

    # reset starting set of points
    start = set(inputs[0].replace('\n','')[inputs[0].index(':')+2:].split(','))

    # reverse the order of the path returned from the findPath function
    path = reversed(findPath(start,end,completed,dependencies,topStart,allTasks))

    # join the reversed path list with commas to print out the full path
    print('4b) path: ',','.join(path))
