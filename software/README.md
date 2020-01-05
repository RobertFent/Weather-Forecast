# Weather-Forecast
Small project for predicting the future weather based on previous data about weather

## How to use this software:
1. Clone this repository
2. Change directory to ./software (terminal-command: cd software)
3. Activate python-env via:
    + Linux/MacOS:
    ```shell
    source python-env/bin/activate
    ```
    + Windows:
    ```shell
    python-env\Scripts\activate.bat
    ```
4. Install all needed libraries via:
```shell
pip install -r requirements.txt
```
5. Set LABEL_NAME to columnname of feature you want to predict
6. Run the script (on linux) via:
```shell
python3 init.py
``` 

### optional: update weather data to learn from:
1. Check if API-key still works (If not: create new [account](https://www.worldweatheronline.com/developer/) and get a new key)
2. Set START_DATE and END_DATE (I prefer using whole months as you can see by the current values of these variables)
3. Set FILE_POST_FIX to desired ending of csv file which will be generated (I prefer using the first three letters of the month)
3. Set LATITUDE and LONGITUDE to desired location
4. Remove '#' from '# fetchDataAndConvert()' in def main()