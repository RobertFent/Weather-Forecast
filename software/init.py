import xml.etree.ElementTree as ET
import csv
import os
import numpy as np
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

API_KEY = '7b1d62fff6684c03b5e150530251305'
BASE_URL = 'https://api.worldweatheronline.com/premium/v1/past-weather.ashx'
START_DATE = '2015-04-01'
END_DATE = '2015-04-30'
FILE_POST_FIX = 'apr_2025'
LATITUDE = '49.013432'
LONGITUDE = '12.101624'

TEMP_COL = 'tempC'
PRECIP_COL = 'precipMM'

# change labelName to change predicted feature
LABEL_NAME = TEMP_COL

# _ needed for string representation
NEXT_DAY = '2019-11-01_'
NEXT_DAY_TIMESTAMPS = [NEXT_DAY + '0', NEXT_DAY + '300', NEXT_DAY + '600', NEXT_DAY +
                       '900', NEXT_DAY + '1200', NEXT_DAY + '1500', NEXT_DAY + '1800', NEXT_DAY + '2100']


def generate_request_url() -> str:
    """Generates a request URL for fetching data based on specified parameters.

    This function constructs a URL by combining a base URL with query parameters 
    including latitude, longitude, start date, end date, and an API key. 

    Returns:
        str: The complete request URL for the API call.
    """
    request_url = BASE_URL + '?' + 'q=' + LATITUDE + ',' + LONGITUDE + '&date=' + \
        START_DATE + '&enddate=' + END_DATE + '&key=' + API_KEY
    return request_url


# https://stackoverflow.com/questions/29810572/save-xml-response-from-get-call-using-python
def parse_response_to_xml_file(res: requests.Response) -> None:
    """Parse the HTTP response and save it as an XML file.

    This function takes an HTTP response object, parses its content as XML, 
    and writes the parsed XML data to a file named 'weatherData.xml'.

    Args:
        res (requests.Response): The HTTP response object containing the XML data.

    Returns:
        None
    """
    print('Parsing response...')
    root = ET.fromstring(res.text)
    tree = ET.ElementTree(root)
    tree.write('weatherData.xml')


# https://www.worldweatheronline.com/developer/api/docs/historical-weather-api.aspx#extraparameter
# 1 weather header per day
# date, astronomy, maxtempC, maxtempF, mintempC, mintempF, acgtempC, avgtempF, totalSnow_cm, sunHour, uvIndex, 8 hourly per weather
# sunrise, sunset, moonrise, moonset, moon_phase, moon_illumination per astronomy (irrelevant)
# time, tempC, tempF, windspeedMiles, windspeedKmph, winddirDegree, winddir16Point, weatherCode, weatherIconUrl, weatherDesc, precipMM, precipInches, humidity, visibility, visibilityMiles, pressure, pressureInches,
# cloudcover, HeatIndexC, HeatIndexF, DewPointC, DewPointF, WindChillC, WindChillF, WindGustMiles, WindGustKmph, FeelsLikeC, FeelsLikeF, uvIndex per hourly

# http://blog.appliedinformaticsinc.com/how-to-parse-and-convert-xml-to-csv-using-python/
def convert_xml_to_csv() -> None:
    """Converts an XML weather data file to a CSV format.

    This function reads weather data from an XML file named 'weatherData.xml',
    extracts relevant information, and writes it to a new CSV file. The CSV file
    is named using a predefined postfix and is saved in the 'weatherData' directory.
    After the conversion, the original XML file is deleted.

    Returns:
        None

    Raises:
        IOError: If there is an issue reading the XML file or writing to the CSV file.
        ET.ParseError: If the XML file is not well-formed.
    """
    print('Converting xml file...')
    new_file_name = f'weatherData/weatherData_{FILE_POST_FIX}.csv'
    with open(new_file_name, 'w', encoding='utf-8') as csv_file, open('weatherData.xml', 'r', encoding='utf-8') as xml_file:
        csv_writer = csv.writer(csv_file)
        csv_headers = []
        root = ET.fromstring(xml_file.read())

        # write csv head general tags
        for elem in root.find('weather'):
            header = elem.tag
            if (header not in ('hourly', 'astronomy')):
                csv_headers.append(header)
        # write csv head hourly tags
        for elem2 in root.find('weather').find('hourly'):
            csv_headers.append(elem2.tag)
        csv_writer.writerow(csv_headers)

        # write csv body
        for elem in root.findall('weather'):
            csv_row_day_data = []
            # iterate over weather elem
            # get day data + append to every hourly data
            for day_data_elem in elem:
                if (day_data_elem.tag != 'astronomy' and day_data_elem.tag != 'hourly'):
                    csv_row_day_data.append(day_data_elem.text)
            # iterate over every hourly elem per weather
            for outer_hourly_data_elem in elem.findall('hourly'):
                csv_row_hourly_data = []
                for inner_hourly_data_elem in outer_hourly_data_elem:
                    csv_row_hourly_data.append(inner_hourly_data_elem.text)
                csv_row = csv_row_day_data + csv_row_hourly_data
                csv_writer.writerow(csv_row)
    # remove not needed file
    os.remove('weatherData.xml')


def get_cleaned_df() -> pd.DataFrame:
    """Retrieves and cleans the weather data from a CSV file.

    This function reads a CSV file containing weather data, drops specified columns that are not needed for analysis, and applies additional data cleaning functions to prepare the DataFrame for further use.

    Returns:
        pandas.DataFrame: A cleaned DataFrame containing the relevant weather data.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        ValueError: If the DataFrame does not contain the expected columns after cleaning.
    """
    dataframe = pd.read_csv(f'weatherData/weatherData_{FILE_POST_FIX}.csv')
    # use the code below instead if more indepth example is needed
    # dataframe = pd.read_csv('weatherData/weatherData_jan_oct.csv')
    # remove fahrenheit|inches|miles cols
    cols_to_drop = ['maxtempF', 'mintempF', 'avgtempF', 'tempF', 'windspeedMiles', 'precipInches',
                    'visibilityMiles', 'pressureInches', 'HeatIndexF', 'DewPointF', 'WindChillF', 'WindGustMiles', 'FeelsLikeF']
    dataframe.drop(cols_to_drop, inplace=True, axis=1)
    # remove icon
    dataframe.drop(['weatherIconUrl'], inplace=True, axis=1)
    modify_date(dataframe)
    remove_non_label_feature_cols(dataframe)
    return dataframe


def modify_date(dataframe: pd.DataFrame) -> None:
    """Modify the 'date' column of a DataFrame by appending corresponding 'time' values.

    This function takes a pandas DataFrame as input, extracts the 'time' and 'date' columns, 
    and modifies the 'date' column by appending the corresponding 'time' values to each date. 
    The 'time' column is then dropped from the DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing 'time' and 'date' columns.

    Returns:
        None: This function modifies the DataFrame in place and does not return a value.
    """
    time_values = []
    date_values = []
    for time in dataframe['time']:
        time_values.append(time)
    for date in dataframe['date']:
        date_values.append(date)
    for x, time in enumerate(time_values):
        date_values[x] += '_' + str(time)

    dataframe['date'] = date_values
    dataframe.drop(['time'], inplace=True, axis=1)


def remove_non_label_feature_cols(dataframe: pd.DataFrame) -> None:
    """Removes non-label feature columns from the given DataFrame.

    This function modifies the input DataFrame by dropping specific columns that are not considered label features. 
    It removes the column at index 6 and a predefined list of columns: 'weatherCode', 'winddir16Point', 
    'FeelsLikeC', and 'weatherDesc'.

    Args:
        dataframe (pd.DataFrame): The DataFrame from which non-label feature columns will be removed.

    Returns:
        None: The function modifies the DataFrame in place and does not return a value.
    """
    # drop 7th col 'uvIndex' -> need to drop it this way because col is duplicate and last one is important
    dataframe.drop(dataframe.columns[6], axis=1, inplace=True)
    # weathercode + winddir16P kind of duplicate                weatherDesc == string
    cols_to_drop = ['weatherCode', 'winddir16Point',
                    'FeelsLikeC', 'weatherDesc']
    dataframe.drop(cols_to_drop, inplace=True, axis=1)


def setup_df(dataframe: pd.DataFrame, label_name: str) -> tuple[pd.DataFrame, int]:
    """Sets up a DataFrame for time series forecasting by creating a label column.

    This function modifies the input DataFrame by setting the index to the 'date' column,
    filling missing values with -99999, and creating a new 'Label' column that contains
    the values of the specified label column shifted by a defined forecast range. 
    Rows with NaN values are then dropped from the DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing time series data.
        label_name (str): The name of the column to be used for labeling.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The modified DataFrame with the 'Label' column.
            - int: The forecast range used for shifting the label.
    """
    # set index of dataframe to date col
    dataframe.set_index('date', inplace=True)
    # setup col for label
    dataframe['Label'] = np.nan
    # setup forecast (8 entries == 1 day)
    forecast_range = 8
    dataframe.fillna(value=-99999, inplace=True)
    dataframe['Label'] = dataframe[label_name].shift(-forecast_range)

    # remove nan rows
    dataframe.dropna(inplace=True)
    return dataframe, forecast_range


def get_sets_from_arrays(x, y) -> list:
    """Splits the input arrays into training and testing sets.

    Args:
        x (array-like): The input features to be split.
        y (array-like): The target variable to be split.

    Returns:
        list: A list containing the training and testing sets for both features and target variable.
               Specifically, it returns (x_train, x_test, y_train, y_test).
    """
    return train_test_split(x, y, test_size=0.2)


def get_arrays(dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Get feature and label arrays from a DataFrame.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame containing features and a label column.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - x (numpy.ndarray): The feature array, with the 'Label' column dropped.
            - y (numpy.ndarray): The label array, containing the values from the 'Label' column.
    """
    x = np.array(dataframe.drop(['Label'], axis=1))
    y = np.array(dataframe['Label'])
    return x, y


def plot_forecast(dataframe: pd.DataFrame, forecast_set: np.ndarray, label_name: str):
    """Plot the weather forecast based on the provided data.

    This function takes a DataFrame and a forecast set, updates the DataFrame with forecast values, and generates a plot visualizing the actual data alongside the forecast.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the historical data.
        forecast_set (np.ndarray): An array of forecasted values to be plotted.
        label_name (str): The name of the column in the DataFrame to be used as the label for the y-axis.

    Returns:
        None: The function saves the plot as a PNG file in the 'forecast' directory.

    Raises:
        FileNotFoundError: If the 'forecast' directory does not exist.
        ValueError: If the length of forecast_set does not match the expected number of forecasted entries.
    """
    style.use('ggplot')
    dataframe['forecast'] = np.nan

    date_iterator = 0
    for i in forecast_set:
        date_iterator += 1
        dataframe.loc[date_iterator] = [
            np.nan for _ in range(len(dataframe.columns)-1)] + [i]

    plt.figure(figsize=(30, 25), dpi=80, facecolor='w', edgecolor='k')
    dataframe[label_name].plot()
    dataframe['forecast'].plot()
    plt.title('Weather Forecast By Robert')
    plt.legend(loc=8)
    plt.xlabel('Date')
    plt.ylabel(label_name)
    plt.savefig(f'forecast/{label_name}_forecast.png')


# inits model and returns trained model
def generate_model(x_train, x_test, y_train, y_test) -> LinearRegression:
    """Generates and trains a Linear Regression model.

    Args:
        x_train (array-like): Training data features.
        x_test (array-like): Test data features.
        y_train (array-like): Training data target values.
        y_test (array-like): Test data target values.

    Returns:
        LinearRegression: The trained Linear Regression model.

    Prints:
        Model-score: The score of the model on the test data.
    """
    model = LinearRegression()
    model.fit(x_train, y_train)
    print('Model-score: ' + str(model.score(x_test, y_test)))
    return model


# returns forecastSet predicted by model
def get_forecast_set(x: np.ndarray, forecast_range: int, model: LinearRegression):
    """Get forecast predictions based on the provided model and input data.

    Args:
        x (np.ndarray): The input data array from which to generate predictions.
        forecast_range (int): The number of data points to use for forecasting.
        model (LinearRegression): The trained linear regression model used for predictions.

    Returns:
        np.ndarray: The predicted values for the specified forecast range.
    """
    x_prediction_set = x[-forecast_range:]
    return model.predict(x_prediction_set)


def fetch_data_and_convert() -> None:
    """Fetches data from a specified URL, parses the response, and converts it to CSV format.

    This function performs the following steps:
    1. Fetches data from a generated request URL.
    2. Checks the response status code; if it is not 200, an error message is printed.
    3. Parses the response and saves it to an XML file.
    4. Converts the XML file to a CSV format.

    Returns:
        None
    """
    print('Fetching data...')
    res = requests.get(generate_request_url(), timeout=5000)
    if (res.status_code != 200):
        print('An error occurred retrieving new weather data\nPlease check the request headers')
        return
    parse_response_to_xml_file(res)
    convert_xml_to_csv()


def main() -> None:
    """Main function to execute the data processing and forecasting workflow.

    This function orchestrates the entire process of fetching data, cleaning it, 
    preparing it for modeling, generating a predictive model, and plotting the 
    forecast results. It calls several helper functions to perform each step 
    in the workflow.

    Steps performed:
    1. Fetch and convert data.
    2. Clean the data and prepare a DataFrame.
    3. Set up the DataFrame for modeling.
    4. Extract feature and target arrays.
    5. Split the arrays into training and testing sets.
    6. Generate a predictive model based on the training data.
    7. Create a forecast set using the model.
    8. Plot the forecast results.

    Returns:
        None
    """
    fetch_data_and_convert()
    df = get_cleaned_df()
    df_ready, forecast_range = setup_df(df, LABEL_NAME)
    x, y = get_arrays(df_ready)
    x_train, x_test, y_train, y_test = get_sets_from_arrays(x, y)

    model = generate_model(x_train, x_test, y_train, y_test)

    forecast_set = get_forecast_set(x, forecast_range, model)

    plot_forecast(df_ready, forecast_set, LABEL_NAME)


main()
