from flask import Flask, jsonify, request
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import subprocess
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import firestore
# voice feature extraction script
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy
import ssl
import urllib.request
import json


app = Flask(__name__)

cred = credentials.Certificate("./admin.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': "finalyearproject-a4622.appspot.com",
})

@app.route('/', methods=['POST'])
def execute_notebook():
    # get uid from post request
    data = request.json
    uid = data.get('uid')
    print("UID form flask = ", uid,data)

    # user info
    # uid = "oCgxzHjizfUCptOGuivj1F1MpU33"
    filename = "Test.wav"

    # This Python 3 environment comes with many helpful analytics libraries installed
    # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
    # For example, here's several helpful packages to load

    # Input data files are available in the read-only "../input/" directory
    # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

        # You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you
        # create a version using "Save & Run All" You can also write temporary files to /kaggle/temp/, but they won't
        # be saved outside of the current session

    # put your key file path

    # Creating a Firestore client
    db = firestore.client()

    bucket = storage.bucket()
    # Get a reference to the file
    blob = storage.bucket().blob(uid + "/" + filename)
    # put your file path here from firebase storage

    # Download the file to a local path, /home/flask/flask_project/audios/uid_filename
    filename_for_audio = f"./Audios/{uid}_{filename}"
    blob.download_to_filename(filename_for_audio)
    #print(filename)

    (rate, sig) = wav.read(filename_for_audio)
    mfcc_feat = mfcc(sig, rate)
    fbank_feat = logfbank(sig, rate)

    a = fbank_feat
    b = mfcc_feat
    c = fbank_feat[1:3, :]  # to store the array of fbank feature
    d = mfcc_feat[1:3, :]  # to store the array of MFCC feature
    # MFCC
    filename_for_csv1 = f"./Csv/VoiceFeatureFbank_{uid}.csv"
    numpy.savetxt(filename_for_csv1, d, delimiter=",")  # Save feature in CSV
    filename_for_csv2 = f"./Csv/VoiceFeatureMFCCFl_{uid}.csv"
    numpy.savetxt(filename_for_csv2, d, delimiter=",")  # Save feature in CSV

    # Get a reference to the destination directory
    destination_blob_name = uid + '/VoiceFeatureFbank.csv'
    blob = storage.bucket().blob(destination_blob_name)

    # Upload the file to Firebase Storage
    blob.upload_from_filename(filename_for_csv1)

    # Get a reference to the destination directory
    destination_blob_name = uid + '/VoiceFeatureMFCCFl.csv'
    blob = storage.bucket().blob(destination_blob_name)

    # Upload the file to Firebase Storage
    blob.upload_from_filename(filename_for_csv2)

    # Script to screen TB/Covid 19
    data = {
        "Inputs": {
            "input1":
                [
                    {
                        'P1': c.item(0),
                        'P2': c.item(1),
                        'P3': c.item(2),
                        'P4': c.item(3),
                        'P5': c.item(4),
                        'P6': c.item(5),
                        'P7': c.item(6),
                        'P8': c.item(7),
                        'P9': c.item(8),
                        'P10': c.item(9),
                        'P11': c.item(10),
                        'P12': c.item(11),
                        'P13': c.item(12),
                        'P14': c.item(13),
                        'P15': c.item(14),
                        'P16': c.item(15),
                        'P17': c.item(16),
                        'P18': c.item(17),
                        'P19': c.item(18),
                        'P20': c.item(19),
                        'P21': c.item(20),
                        'P22': c.item(21),
                        'P23': c.item(22),
                        'P24': c.item(23),
                        'P25': c.item(24),
                        'P26': c.item(25),
                        'P27': c.item(26),
                        'P28': c.item(27),
                        'P29': c.item(28),
                        'P30': c.item(29),
                        'P31': c.item(30),
                        'P32': c.item(31),
                        'P33': c.item(32),
                        'P34': c.item(33),
                        'P35': c.item(34),
                        'P36': c.item(35),
                        'P37': c.item(36),
                        'P38': c.item(37),
                        'P39': c.item(38),
                        'P40': c.item(39),
                        'P41': c.item(40),
                        'P42': c.item(41),
                        'P43': c.item(42),
                        'P44': c.item(43),
                        'P45': c.item(44),
                        'P46': c.item(45),
                        'P47': c.item(46),
                        'P48': c.item(47),
                        'P49': c.item(48),
                        'P50': c.item(49),
                        'P51': c.item(50),
                        'P52': c.item(51),
                        'P53': d.item(0),
                        'P54': d.item(1),
                        'P55': d.item(2),
                        'P56': d.item(3),
                        'P57': d.item(4),
                        'P58': d.item(5),
                        'P59': d.item(6),
                        'P60': d.item(7),
                        'P61': d.item(8),
                        'P62': d.item(9),
                        'P63': d.item(10),
                        'P64': d.item(11),
                        'P65': d.item(12),
                        'P66': d.item(13),
                        'P67': d.item(14),
                        'P68': d.item(15),
                        'P69': d.item(16),
                        'P70': d.item(17),
                        'P71': d.item(18),
                        'P72': d.item(19),
                        'P73': d.item(20),
                        'P74': d.item(21),
                        'P75': d.item(22),
                        'P76': d.item(23),
                        'P77': d.item(24),
                        'P78': d.item(25),
                        'P79': "150",
                        'PID': "33",
                        'Age': "278",
                        'Gender': "0",
                        'Result': "1",
                    }
                ],
        },
        "GlobalParameters": {
        }
    }

    body = str.encode(json.dumps(data))

    url = 'https://ussouthcentral.services.azureml.net/workspaces/bd0a45edaff24523bc514f868bcbe3b0/services/7cfbd9848b2c47178962d99df38f34ed/execute?api-version=2.0&format=swagger'
    api_key = 'BdFPTFJ7m8MM+tt016wb8yzzwTwMapWDUqqOh0bDltT+p5sqBuwYVl6LVEyM7dyQM3LVEKOFqh71UkwM88TbUA=='  # Replace this with the API key for the web service
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

    ssl._create_default_https_context = ssl._create_unverified_context
    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()

        #     for formatted json
        parsed_result = json.loads(result)
        formatted_result = json.dumps(parsed_result, indent=4)
        # print(formatted_result)

    #     print(result)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))

    covid19_tb = parsed_result["Results"]["output1"][0]["Scored Probabilities"]
    print(covid19_tb)

    if (result[2066:2067]) == b'0':
        print("The persone is Covid 19 Negative")
    if (result[2066:2067]) == b'1':
        print("The persone is Covid 19 Positive")

    # Script to screen Cough and Cold
    data = {
        "Inputs": {
        },
        "GlobalParameters": {
        }
    }

    body = str.encode(json.dumps(data))

    url = 'https://ussouthcentral.services.azureml.net/workspaces/bd0a45edaff24523bc514f868bcbe3b0/services/f2c6c1a06d7f4d2c9738b7b2a29db11a/execute?api-version=2.0&format=swagger'
    api_key = '5E8K93yZmnKRJKR1vvcQfmjitXYt1Y3Z2AAFEPyS3AGEYdCPLizzvjwnorLQE+qwA9v5QU08BJtbxoHkHzsVIQ=='  # Replace this with the API key for the web service
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()

        #     for formatted json
        parsed_result = json.loads(result)
        formatted_result = json.dumps(parsed_result, indent=4)
        # print(formatted_result)

    #     print(result)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))

    # Script to screen diabetes
    data = {
        "Inputs": {
            "input1":
                [
                    {
                        'P1': "229.12",
                        'P2': "12.1",
                        'P3': "12.45",
                        'P4': "229.12",
                        'P5': "12.1",
                        'P6': "12.45",
                        'P7': "229.12",
                        'P8': "12.1",
                        'P9': "12.45",
                        'P10': "229.12",
                        'P11': "12.1",
                        'P12': "12.45",
                        'P13': "229.12",
                        'P14': "12.1",
                        'P15': "12.45",
                        'P16': "229.12",
                        'P17': "12.1",
                        'P18': "12.45",
                        'P19': "229.12",
                        'P20': "12.1",
                        'P21': "12.45",
                        'P22': "229.12",
                        'P23': "12.1",
                        'P24': "12.45",
                        'P25': "229.12",
                        'P26': "12.1",
                        'P27': "12.45",
                        'P28': "229.12",
                        'P29': "12.1",
                        'P30': "12.45",
                        'P31': "229.12",
                        'P32': "12.1",
                        'P33': "12.45",
                        'P34': "229.12",
                        'P35': "12.1",
                        'P36': "12.45",
                        'P37': "229.12",
                        'P38': "12.1",
                        'P39': "12.45",
                        'P40': "229.12",
                        'P41': "12.1",
                        'P42': "12.45",
                        'P43': "229.12",
                        'P44': "12.1",
                        'P45': "12.45",
                        'P46': "229.12",
                        'P47': "12.1",
                        'P48': "12.45",
                        'P49': "229.12",
                        'P50': "12.1",
                        'P51': "12.45",
                        'P52': "229.12",
                        'P53': "12.1",
                        'P54': "12.45",
                        'P55': "229.12",
                        'P56': "12.1",
                        'P57': "12.45",
                        'P58': "229.12",
                        'P59': "12.1",
                        'P60': "12.45",
                        'P61': "229.12",
                        'P62': "12.1",
                        'P63': "12.45",
                        'P64': "229.12",
                        'P65': "12.12",
                        'P66': "12.45",
                        'P67': "229.12",
                        'P68': "12.1",
                        'P69': "12.45",
                        'P70': "229.12",
                        'P71': "12.1",
                        'P72': "12.45",
                        'P73': "123.44",
                        'P74': "324.32",
                        'P75': "324.23",
                        'P76': "32432.32",
                        'P77': "324.34",
                        'P78': "343.34",
                        'P79': "43.43",
                        'PID': "33",
                        'Age': "278",
                        'Gender': "1",
                        'Result': "1",
                    }
                ],
        },
        "GlobalParameters": {
        }
    }

    body = str.encode(json.dumps(data))

    url = 'https://ussouthcentral.services.azureml.net/workspaces/bd0a45edaff24523bc514f868bcbe3b0/services/f383aaaa58a446ecacccbb3841ac7750/execute?api-version=2.0&format=swagger'
    api_key = 'DhhS3hLfci3Y/u3alOKrP1yFdSG3VjeGgc57roXfbXUz0heRu5KWxOPmSgHpZ9gIlIHqFs6XYpEKRWiw6dmlMA=='  # Replace this with the API key for the web service
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

    ssl._create_default_https_context = ssl._create_unverified_context
    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()

        #     for formatted json
        parsed_result = json.loads(result)
        formatted_result = json.dumps(parsed_result, indent=4)
        # print(formatted_result)

    #     print(result)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))

    diabetes = parsed_result["Results"]["output1"][0]["Scored Probabilities"]
    print(diabetes)

    # Script to screen AML

    data = {
        "Inputs": {
            "input1":
                [
                    {
                        'P1': "229",
                        'P2': "124",
                        'P3': "221",
                        'P4': "80",
                        'P5': "101",
                        'P6': "225",
                        'P7': "13",
                        'P8': "215",
                        'P9': "178",
                        'P10': "70",
                        'P11': "105",
                        'P12': "272",
                        'P13': "105",
                        'P14': "161",
                        'P15': "287",
                        'P16': "104",
                        'P17': "248",
                        'P18': "112",
                        'P19': "215",
                        'P20': "154",
                        'P21': "230",
                        'P22': "211",
                        'P23': "123",
                        'P24': "264",
                        'P25': "99",
                        'P26': "287",
                        'P27': "233",
                        'P28': "77",
                        'P29': "150",
                        'P30': "33",
                        'P31': "278",
                        'P32': "295",
                        'P33': "75",
                        'P34': "68",
                        'P35': "56",
                        'P36': "208",
                        'P37': "267",
                        'P38': "88",
                        'P39': "225",
                        'P40': "273",
                        'P41': "208",
                        'P42': "292",
                        'P43': "172",
                        'P44': "69",
                        'P45': "199",
                        'P46': "272",
                        'P47': "61",
                        'P48': "236",
                        'P49': "217",
                        'P50': "120",
                        'P51': "229",
                        'P52': "124",
                        'P53': "221",
                        'P54': "80",
                        'P55': "101",
                        'P56': "225",
                        'P57': "13",
                        'P58': "215",
                        'P59': "178",
                        'P60': "70",
                        'P61': "105",
                        'P62': "272",
                        'P63': "105",
                        'P64': "161",
                        'P65': "287",
                        'P66': "104",
                        'P67': "248",
                        'P68': "112",
                        'P69': "215",
                        'P70': "154",
                        'P71': "230",
                        'P72': "211",
                        'P73': "123",
                        'P74': "264",
                        'P75': "99",
                        'P76': "287",
                        'P77': "233",
                        'P78': "77",
                        'P79': "150",
                        'PID': "33",
                        'Age': "278",
                        'Gender': "295",
                        'Result': "1",
                    }
                ],
        },
        "GlobalParameters": {
        }
    }

    body = str.encode(json.dumps(data))

    url = 'https://ussouthcentral.services.azureml.net/workspaces/bd0a45edaff24523bc514f868bcbe3b0/services/7cfbd9848b2c47178962d99df38f34ed/execute?api-version=2.0&format=swagger'
    api_key = 'BdFPTFJ7m8MM+tt016wb8yzzwTwMapWDUqqOh0bDltT+p5sqBuwYVl6LVEyM7dyQM3LVEKOFqh71UkwM88TbUA=='  # Replace this with the API key for the web service
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

    ssl._create_default_https_context = ssl._create_unverified_context
    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()

        #     for formatted json
        parsed_result = json.loads(result)
        formatted_result = json.dumps(parsed_result, indent=4)
        # print(formatted_result)

    #     print(result)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))

    aml = parsed_result["Results"]["output1"][0]["Scored Probabilities"]
    print(aml)

    # Script to screen Mental Health

    data = {
        "Inputs": {
            "input1":
                [
                    {
                        'P1': c.item(0),
                        'P2': c.item(1),
                        'P3': c.item(2),
                        'P4': c.item(3),
                        'P5': c.item(4),
                        'P6': c.item(5),
                        'P7': c.item(6),
                        'P8': c.item(7),
                        'P9': c.item(8),
                        'P10': c.item(9),
                        'P11': c.item(10),
                        'P12': c.item(11),
                        'P13': c.item(12),
                        'P14': c.item(13),
                        'P15': c.item(14),
                        'P16': c.item(15),
                        'P17': c.item(16),
                        'P18': c.item(17),
                        'P19': c.item(18),
                        'P20': c.item(19),
                        'P21': c.item(20),
                        'P22': c.item(21),
                        'P23': c.item(22),
                        'P24': c.item(23),
                        'P25': c.item(24),
                        'P26': c.item(25),
                        'P27': c.item(26),
                        'P28': c.item(27),
                        'P29': c.item(28),
                        'P30': c.item(29),
                        'P31': c.item(30),
                        'P32': c.item(31),
                        'P33': c.item(32),
                        'P34': c.item(33),
                        'P35': c.item(34),
                        'P36': c.item(35),
                        'P37': c.item(36),
                        'P38': c.item(37),
                        'P39': c.item(38),
                        'P40': c.item(39),
                        'P41': c.item(40),
                        'P42': c.item(41),
                        'P43': c.item(42),
                        'P44': c.item(43),
                        'P45': c.item(44),
                        'P46': c.item(45),
                        'P47': c.item(46),
                        'P48': c.item(47),
                        'P49': c.item(48),
                        'P50': c.item(49),
                        'P51': c.item(50),
                        'P52': c.item(51),
                        'P53': d.item(0),
                        'P54': d.item(1),
                        'P55': d.item(2),
                        'P56': d.item(3),
                        'P57': d.item(4),
                        'P58': d.item(5),
                        'P59': d.item(6),
                        'P60': d.item(7),
                        'P61': d.item(8),
                        'P62': d.item(9),
                        'P63': d.item(10),
                        'P64': d.item(11),
                        'P65': d.item(12),
                        'P66': d.item(13),
                        'P67': d.item(14),
                        'P68': d.item(15),
                        'P69': d.item(16),
                        'P70': d.item(17),
                        'P71': d.item(18),
                        'P72': d.item(19),
                        'P73': d.item(20),
                        'P74': d.item(21),
                        'P75': d.item(22),
                        'P76': d.item(23),
                        'P77': d.item(24),
                        'P78': d.item(25),
                        'P79': "150",
                        'PID': "33",
                        'Age': "278",
                        'Gender': "0",
                        'Result': "1",
                    }
                ],
        },
        "GlobalParameters":  {
        }
    }

    body = str.encode(json.dumps(data))

    url = 'https://ussouthcentral.services.azureml.net/workspaces/bd0a45edaff24523bc514f868bcbe3b0/services/7cfbd9848b2c47178962d99df38f34ed/execute?api-version=2.0&format=swagger'
    api_key = 'BdFPTFJ7m8MM+tt016wb8yzzwTwMapWDUqqOh0bDltT+p5sqBuwYVl6LVEyM7dyQM3LVEKOFqh71UkwM88TbUA=='# Replace this with the API key for the web service
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    ssl._create_default_https_context = ssl._create_unverified_context
    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()

        #     for formatted json
        parsed_result = json.loads(result)
        formatted_result = json.dumps(parsed_result, indent=4)
        # print(formatted_result)

    #     print(result)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))

    mental_health = parsed_result["Results"]["output1"][0]["Scored Probabilities"]
    print(mental_health)

    # Script to screen cardiac health

    data = {
        "Inputs": {
            "input1":
                [
                    {
                        'P1': "229",
                        'P2': "124",
                        'P3': "221",
                        'P4': "80",
                        'P5': "101",
                        'P6': "225",
                        'P7': "13",
                        'P8': "215",
                        'P9': "178",
                        'P10': "70",
                        'P11': "105",
                        'P12': "272",
                        'P13': "105",
                        'P14': "161",
                        'P15': "287",
                        'P16': "104",
                        'P17': "248",
                        'P18': "112",
                        'P19': "215",
                        'P20': "154",
                        'P21': "230",
                        'P22': "211",
                        'P23': "123",
                        'P24': "264",
                        'P25': "99",
                        'P26': "287",
                        'P27': "233",
                        'P28': "77",
                        'P29': "150",
                        'P30': "33",
                        'P31': "278",
                        'P32': "295",
                        'P33': "75",
                        'P34': "68",
                        'P35': "56",
                        'P36': "208",
                        'P37': "267",
                        'P38': "88",
                        'P39': "225",
                        'P40': "273",
                        'P41': "208",
                        'P42': "292",
                        'P43': "172",
                        'P44': "69",
                        'P45': "199",
                        'P46': "272",
                        'P47': "61",
                        'P48': "236",
                        'P49': "217",
                        'P50': "120",
                        'P51': "229",
                        'P52': "124",
                        'P53': "221",
                        'P54': "80",
                        'P55': "101",
                        'P56': "225",
                        'P57': "13",
                        'P58': "215",
                        'P59': "178",
                        'P60': "70",
                        'P61': "105",
                        'P62': "272",
                        'P63': "105",
                        'P64': "161",
                        'P65': "287",
                        'P66': "104",
                        'P67': "248",
                        'P68': "112",
                        'P69': "215",
                        'P70': "154",
                        'P71': "230",
                        'P72': "211",
                        'P73': "123",
                        'P74': "264",
                        'P75': "99",
                        'P76': "287",
                        'P77': "233",
                        'P78': "77",
                        'P79': "150",
                        'PID': "33",
                        'Age': "278",
                        'Gender': "295",
                        'Result': "1",
                    }
                ],
        },
        "GlobalParameters":  {
        }
    }

    body = str.encode(json.dumps(data))

    url = 'https://ussouthcentral.services.azureml.net/workspaces/bd0a45edaff24523bc514f868bcbe3b0/services/7cfbd9848b2c47178962d99df38f34ed/execute?api-version=2.0&format=swagger'
    api_key = 'BdFPTFJ7m8MM+tt016wb8yzzwTwMapWDUqqOh0bDltT+p5sqBuwYVl6LVEyM7dyQM3LVEKOFqh71UkwM88TbUA=='# Replace this with the API key for the web service
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    ssl._create_default_https_context = ssl._create_unverified_context
    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()

        #     for formatted json
        parsed_result = json.loads(result)
        formatted_result = json.dumps(parsed_result, indent=4)
        # print(formatted_result)

    #     print(result)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))

    cardiac_health = parsed_result["Results"]["output1"][0]["Scored Probabilities"]
    print(cardiac_health)

        # Script to screen Cough and Cold 2

    data = {
        "Inputs": {
            "input1":
                [
                    {
                        'P1': "164.592",
                        'P2': "160.258",
                        'P3': "17.112",
                        'P4': "74.855",
                        'P5': "213.894",
                        'P6': "668",
                        'P7': "649",
                        'P8': "11.479",
                        'P9': "12",
                        'P10': "18.541",
                        'P11': "0.02842",
                        'P12': "0.0147",
                        'P13': "0.0146",
                        'P14': "0.044",
                        'P15': "0.0779",
                        'P16': "0.893",
                        'P17': "0.03681",
                        'P18': "0.0513",
                        'P19': "0.0747",
                        'P20': "0.1104",
                        'P21': "0.893992",
                        'P22': "0.169401",
                        'P23': "14.56",
                        'P24': "1",
                        'P25': "2",
                        'P26': "1",
                        'Diseases Condition': "Cough and cold",
                    }
                ],
        },
        "GlobalParameters":  {
        }
    }

    body = str.encode(json.dumps(data))

    url = 'https://ussouthcentral.services.azureml.net/workspaces/bd0a45edaff24523bc514f868bcbe3b0/services/3e63012171cc4c37bc546dc6fab2c2a7/execute?api-version=2.0&format=swagger'
    api_key = 'qrrAuQRcJMwllYA4BPkDrhlyAZmEC89ErQ4UWWEmsN+ypx4yeg1S+5rP3bje+FnkYEL5W3wCfijyodno6XsX1A==' # Replace this with the API key for the web service
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    ssl._create_default_https_context = ssl._create_unverified_context
    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()

        #     for formatted json
        parsed_result = json.loads(result)
        formatted_result = json.dumps(parsed_result, indent=4)
        # print(formatted_result)

    #     print(result)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))

    cold_cough2 = parsed_result["Results"]["output1"][0]["Scored Probabilities"]
    print(cold_cough2)

    print("Scored Probobility of Covid19 & TB is ", covid19_tb)
    print("Scored Probobility of Diabetes is ", diabetes)
    print("Scored Probobility of AML is ", aml)
    print("Scored Probobility of Mental Health is ", mental_health)
    print("Scored Probobility of Cardiac Health is ", cardiac_health)
    print("Scored Probobility of Cold & Cough2 is ", cold_cough2)

    db = firestore.client()

    def get_user_doc(uid):
        user_ref = db.collection('users').document(uid)
        user_doc = user_ref.get()
        return user_doc

    def upload_data(uid, data):
        user_ref = db.collection('users').document(uid)
        user_ref.update(data)
        print("Data uploaded successsfully!")

    # uid = "oCgxzHjizfUCptOGuivj1F1MpU33"
    user_doc = get_user_doc(uid)
    
    data = {
            "covid19_tb" : covid19_tb,
            "diabetes" : diabetes,
            "aml" : aml,
            "mental_health" : mental_health,
            "cardiac_health" : cardiac_health,
            "cold_cough2" : cold_cough2
        }
        # upload_data(uid, data)
    return data
   


@app.route('/ping', methods=['GET'])
def hello():
    return "Hello jfufufuf"


if __name__ == '__main__':
    app.run()
