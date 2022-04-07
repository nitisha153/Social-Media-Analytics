"""
Social Media Analytics Project
Name:
Roll Number:
"""

import hw6_social_tests as test

project = "Social" # don't edit this

### PART 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]

'''
makeDataFrame(filename)
#3 [Check6-1]
Parameters: str
Returns: dataframe
'''
def makeDataFrame(filename):
    df = pd.read_csv(filename)
    return df
'''
parseName(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseName(fromString):
    name = fromString 
    name_label = name.split(':')[1].split('(')[0]
       
    #print(str(name_label)) 

    return name_label.strip()


'''
parsePosition(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parsePosition(fromString):
    position = fromString
    position_label = position.split(':')[1].split('(')[1].split('from')[0]
    #print(str(position_label.strip()))
    return str(position_label.strip())


'''
parseState(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseState(fromString):
    state = fromString
    state_label = state.split(':')[1].split('(')[1].split(')')[0].split('from')[1]
    #print(state_label)
    return state_label.strip()


'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    msg = message
    hashtags = []
    string = ""
    tags = msg.split('#')
    for index in range(1,len(tags)):
        #print(word)
        #i = 0
        for char in tags[index]:
            if char not in endChars:
                string += char
                #print(string)
                #i += 1
            else:
                break
        string = "#" + string
        hashtags.append(string)
        string =""
    # for word in msg.split():
    #     if word[0] == '#':
    #         #print(word[0],word)
    #         string = "#"
    #         i = 1
    #         while word[i] not in endChars:
    #             string += word[i]
    #             #print(word[i],i)
    #             i += 1
    #             if i == len(word):
    #                 break

    #         hashtags.append(string)
    #print(hashtags)
    return hashtags


'''
getRegionFromState(stateDf, state)
#6 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    find_region = stateDf.loc[stateDf["state"] == state,'region']
    #print(find_region)
    return find_region.values[0]


'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    names = []
    positions = []
    states = []
    regions = []
    hashtags = []

    for index,row in data.iterrows():
        value = row['label']
        name = parseName(value)
        names.append(name)
        position = parsePosition(value)
        positions.append(position)
        state = parseState(value)
        states.append(state)
        region = getRegionFromState(stateDf, state)
        regions.append(region)
        text_value = row['text']
        hashtag = findHashtags(text_value)
        hashtags.append(hashtag)
    data['name'] = names
    data['position'] = positions
    data['state'] = states
    data['region'] = regions
    data['hashtags'] = hashtags
    return None


### PART 2 ###

'''
findSentiment(classifier, message)
#1 [Check6-2]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if score < -0.1:
        return "negative"
    elif score > 0.1:
        return "positive"
    else:
        return "neutral"


'''
addSentimentColumn(data)
#2 [Check6-2]
Parameters: dataframe
Returns: None
'''
def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    sentiments = []
    for index,row in data.iterrows():
        message = row['text']
        sentiment = findSentiment(classifier,message)
        sentiments.append(sentiment)
    data['sentiment'] = sentiments
    return None


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    dict_state = {}
    if colName != "" and dataToCount != "":
        for index, row in data.iterrows():
            state = row['state']
            col = row[colName]
            if col == dataToCount:
                if state not in dict_state:
                    dict_state[state] = 0
                dict_state[state] += 1
    else:
        for index, row in data.iterrows():
            state = row['state']
            if state not in dict_state:
                dict_state[state] = 0
            dict_state[state] += 1
    return dict_state


'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    d = {}
    for index, row in data.iterrows():
        key = row['region']
        if key not in d:
            d[key] = {}
        if row[colName] not in d[key]:
            d[key][row[colName]] = 0
        d[key][row[colName]] += 1

    
    return d


'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    tag_counts = {}
    for index,row in data.iterrows():
        lst = row['hashtags']
        for tag in lst:
            if tag not in tag_counts:
                tag_counts[tag] = 0
            tag_counts[tag] += 1

    return tag_counts


'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    # import heapq
    mostcommon_tags = {}
    # lst = heapq.nlargest(count,hashtags,key=hashtags.get)
    # #print(lst)
    # for tag in lst:
    #     if hashtags[tag] not in mostcommon_tags:
    #         mostcommon_tags[tag] = hashtags[tag]
    d_descending = list(sorted(hashtags.items(), 
                                  key=lambda kv: kv[1], reverse=True))
    #print(d_descending)
    for index in range(len(d_descending)):
        if index <= count - 1:
            mostcommon_tags[d_descending[index][0]] = d_descending[index][1]
    
    return mostcommon_tags


'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    sum_sentiments = 0
    total = 0
    for index, row in data.iterrows():
        msg = row['text']
        if hashtag in msg:
            total +=1
            if row['sentiment'] == 'positive':
                sum_sentiments += 1
            if row['sentiment'] == 'negative':
                sum_sentiments -= 1
    return sum_sentiments/total


### PART 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    keys = list(stateCounts.keys())
    values = list(stateCounts.values())
    plt.bar(keys,values,width=0.9)
    plt.xticks(rotation="vertical")
    plt.xlabel("State Names")
    plt.ylabel("No.of Messages per state")
    plt.title(title)
    plt.show()
    return None


'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    feature_rates ={}
    for state in stateCounts:
        if state in stateFeatureCounts:
            frequency_rate = stateFeatureCounts[state]/stateCounts[state]
            feature_rates[state] = frequency_rate
    #to get top N states by using function mostCommonHashtags
    top_N_states = mostCommonHashtags(feature_rates,n)
    graphStateCounts(top_N_states,title) 

    return None


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    features = []
    regions = []
    region_features = []
    for region in regionDicts:
        for feature in regionDicts[region]:
            if feature not in features:
                features.append(feature)
    
    for region in regionDicts.keys():
        regions.append(region)
    
    for regions in regionDicts:
        temp_region = []
        for feature in features:
            if feature in regionDicts[region]:
                count = regionDicts[region][feature]
                temp_region.append(count)
            else:
                temp_region = 0
        region_features.append(temp_region)
    
    sideBySideBarPlots(features,regions,region_features,title)




    return None


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    return


#### PART 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    # print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    # test.week1Tests()
    # print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    #test.runWeek1()
    # test.testMakeDataFrame()
    # test.testParseName()
    # test.testParsePosition()
    # test.testParseState()
    # test.testFindHashtags()
    # test.testGetRegionFromState()
    # test.testAddColumns()
    # test.testFindSentiment()
    # test.testAddSentimentColumn()
    # test.testGetDataCountByState()
    ## Uncomment these for Week 2 ##
    """print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()"""

    ## Uncomment these for Week 3 ##
    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
