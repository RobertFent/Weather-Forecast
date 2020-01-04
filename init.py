import numpy as np
import requests
import xml.etree.ElementTree as ET
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

API_KEY = 'e687fa98d40547f3ab690449192311'
BASE_URL = 'https://api.worldweatheronline.com/premium/v1/past-weather.ashx'
START_DATE = '2019-10-01'
END_DATE = '2019-10-31'
TEMP_COL = 'tempC'
PRECIP_COL = 'precipMM'

# _ needed for string representation
NEXT_DAY = '2019-11-01_'
NEXT_DAY_TIMESTAMPS = [NEXT_DAY + '0', NEXT_DAY +  '300', NEXT_DAY +  '600', NEXT_DAY + '900', NEXT_DAY + '1200', NEXT_DAY + '1500', NEXT_DAY + '1800', NEXT_DAY + '2100']


def generateRequestURL():
    request_url = BASE_URL + '?' + 'q=49.013432,12.101624&date=' + \
        START_DATE + '&enddate=' + END_DATE + '&key=' + API_KEY
    return request_url


# https://stackoverflow.com/questions/29810572/save-xml-response-from-get-call-using-python
def parseResponseToXMLFile(res):
    root = ET.fromstring(res.text)
    tree = ET.ElementTree(root)
    tree.write("weatherData.xml")


# https://www.worldweatheronline.com/developer/api/docs/historical-weather-api.aspx#extraparameter
# 1 weather header per day
# date, astronomy, maxtempC, maxtempF, mintempC, mintempF, acgtempC, avgtempF, totalSnow_cm, sunHour, uvIndex, 8 hourly per weather
# sunrise, sunset, moonrise, moonset, moon_phase, moon_illumination per astronomy (irrelevant)
# time, tempC, tempF, windspeedMiles, windspeedKmph, winddirDegree, winddir16Point, weatherCode, weatherIconUrl, weatherDesc, precipMM, precipInches, humidity, visibility, visibilityMiles, pressure, pressureInches,
# cloudcover, HeatIndexC, HeatIndexF, DewPointC, DewPointF, WindChillC, WindChillF, WindGustMiles, WindGustKmph, FeelsLikeC, FeelsLikeF, uvIndex per hourly

# http://blog.appliedinformaticsinc.com/how-to-parse-and-convert-xml-to-csv-using-python/
def convertXMLToCSV():

    with open('weatherData/weatherData_nov.csv', 'w') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvHeaders = []
        with open("weatherData.xml", "r") as xmlFile:
            root = ET.fromstring(xmlFile.read())
            # write csv head general tags
            for elem in root.find('weather'):
                header = elem.tag
                if (header != 'hourly' and header != 'astronomy'):
                    csvHeaders.append(header)
            # write csv head hourly tags
            for elem2 in root.find('weather').find('hourly'):
                csvHeaders.append(elem2.tag)
            csvWriter.writerow(csvHeaders)

            # write csv body
            for elem in root.findall('weather'):
                csvRowDayData = []
                # iterate over weather elem
                # get day data + append to every hourly data
                for dayDataElem in elem:
                    if (dayDataElem.tag != 'astronomy' and dayDataElem.tag != 'hourly'):
                        csvRowDayData.append(dayDataElem.text)
                # iterate over every hourly elem per weather
                for outerHourlyDataElem in elem.findall('hourly'):
                    csvRowHourlyData = []
                    for innerHourlyDataElem in outerHourlyDataElem:
                        csvRowHourlyData.append(innerHourlyDataElem.text)
                    csvRow = csvRowDayData + csvRowHourlyData
                    csvWriter.writerow(csvRow)

# returns dataframe with only needed cols
def getCleanedDf():
    dataframe = pd.read_csv('weatherData/weatherData_jan_oct.csv')
    # remove fahrenheit|inches|miles cols
    colsToDrop = ['maxtempF', 'mintempF', 'avgtempF', 'tempF', 'windspeedMiles', 'precipInches', 'visibilityMiles', 'pressureInches', 'HeatIndexF', 'DewPointF', 'WindChillF', 'WindGustMiles', 'FeelsLikeF']
    dataframe.drop(colsToDrop, inplace=True, axis=1)
    # remove icon
    dataframe.drop(['weatherIconUrl'], inplace=True, axis=1)
    modifyDate(dataframe)
    removeNonLabelFeatureCols(dataframe)
    return dataframe

# removes col for time and adds time to datestring
def modifyDate(dataframe):
    timeValues = []
    dateValues = []
    for time in dataframe['time']:
        timeValues.append(time)
    for date in dataframe['date']:
        dateValues.append(date)
    for x in range(0, len(timeValues)):
        dateValues[x] += '_' + str(timeValues[x])
    
    dataframe['date'] = dateValues
    dataframe.drop(['time'], inplace=True, axis=1)

# removes cols from df which are neither features nor labels
def removeNonLabelFeatureCols(dataframe):
    # drop 7th col 'uvIndex' -> need to drop it this way because col is duplicate and last one is important
    dataframe.drop(dataframe.columns[6], axis=1, inplace=True)
    # weathercode + winddir16P kind of duplicate                weatherDesc == string
    colsToDrop = ['weatherCode', 'winddir16Point', 'FeelsLikeC', 'weatherDesc']
    dataframe.drop(colsToDrop, inplace=True, axis=1)

# prepares df for ml process
def setupDf(dataframe, labelName):
    # set index of dataframe to date col
    dataframe.set_index('date', inplace=True)
    # setup col for label
    dataframe['Label'] = np.nan
    # setup forecast (8 entries == 1 day)
    forecastRange = 8
    dataframe.fillna(value = -99999, inplace=True)
    dataframe['Label'] = dataframe[labelName].shift(-forecastRange)

    # remove nan rows
    dataframe.dropna(inplace=True)
    return dataframe, forecastRange


# splits df into test + train set (80%train, 20%test)
# returns sets
def splitDataFrame(dataframe, X, y):
    return train_test_split(X, y, test_size=0.2)


# returns x +y array
def getArrays(dataframe):
    X = np.array(dataframe.drop(['Label'], 1))
    y = np.array(dataframe['Label'])
    return X, y


# x-axis: date
# y-axis temp
# plots the dataframe + forecast
def plotForecast(dataframe, forecastSet, labelName):
    style.use('ggplot')
    dataframe['forecast'] = np.nan
    
    dateIterator = 0
    for i in forecastSet:
        dateIterator += 1
        dataframe.loc[dateIterator] = [np.nan for _ in range(len(dataframe.columns)-1)] + [i]

    fig = plt.figure(figsize=(30,25), dpi=80, facecolor='w', edgecolor='k')
    dataframe[labelName].plot()
    dataframe['forecast'].plot()
    plt.title('Weather Forecast By Robert')
    plt.legend(loc=8)
    plt.xlabel('Date')
    plt.ylabel(labelName)
    plt.savefig('%s_forecast.png' % (labelName))


# inits model and returns trained model
def generateModel(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    print('Model-score: ' + str(model.score(X_test, y_test)))
    return model
    

# returns forecastSet predicted by model
def getForecastSet(X, forecastRange, model):
    X_predictionSet = X[-forecastRange:]
    return model.predict(X_predictionSet)


# gets weatherdata from api and parses it to csv
def fetchDataAndConvert():
    res = requests.get(generateRequestURL())
    parseResponseToXMLFile(res)
    convertXMLToCSV()

def main():

    # fetchDataAndConvert()

    labelName = TEMP_COL

    df = getCleanedDf()
    dfReady, forecastRange = setupDf(df, labelName)
    X, y = getArrays(dfReady)
    X_train, X_test, y_train, y_test = splitDataFrame(dfReady, X, y)
    
    model = generateModel(X_train, X_test, y_train, y_test)

    forecastSet = getForecastSet(X, forecastRange, model)

    plotForecast(dfReady, forecastSet, labelName)


main()
