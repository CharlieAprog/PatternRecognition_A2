# PatternRecognition_A2
Repository for the second assignment of the Pattern Recognition course 2021/2022, Group 21.


## Installing the Requirements
To install the requirements to make sure the code will run as expected,
>run: *pip install -r requirements.txt*

## Running the Files
The individual files from be launched from the src directory
to run a pipeline (eg. numeric data pipeline)
*Note, the image script is in the form of a jyupiter notebook, here you must also be cd'd into src to make sure the paths work*
>*cd src*
>
>*python3 numeric.py*
>*python3 semi-supervised.py*

## Reading the data
Since pusing csv files that are over 100mb is not the best practice, the data of the numeric and the semi-supervised sections are not there
To run them, please insert the files, in the state they were presented in the assignment folder into the following directories:
(*note: this does not apply for the big cats data since that data is only small and has been pushed*)

├─ data
│   ├─ BigCats
│   │   ├─ Cheetah
│   │   ├─ Jaguar
│   │   ├─ Leopard
│   │   └─ Lion
│   ├─ CreditCards
│   │   └─ creditcard.csv
│   └─ Genes
│       ├─ data.csv
│       └─ labels.csv
