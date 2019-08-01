from __future__ import print_function, unicode_literals
import datetime
import glob
import sys
import joblib
import zipcodes
import regex
import os
from pprint import pprint

from PyInquirer import style_from_dict, Token, prompt, Separator
from PyInquirer import Validator, ValidationError

import delivery_prediction_predict
import paths

from examples import custom_style_2

class DateValidator(Validator):
    '''
    abc
    '''
    def validate(self, document):
        try:
            datetime.datetime.strptime(document.text, '%Y-%m-%d')
        except ValueError:
            raise ValidationError(
                message='Please enter a valid date in YYYY-MM-DD format',
                cursor_position=len(document.text))  # Move cursor to end

class ZipcodeValidator(Validator):
    def validate(self, document):
        ok = False
        if document.text.isdigit() and len(document.text) == 5:
            ok = zipcodes.is_real(document.text)
        if not ok:
            raise ValidationError(
                message='Please enter a valid zipcode',
                cursor_position=len(document.text))  # Move cursor to end


class NumberValidator(Validator):
    def validate(self, document):
        try:
            float(document.text)
        except ValueError:
            raise ValidationError(
                message='Please enter a number',
                cursor_position=len(document.text))  # Move cursor to end


class WeightValidator(Validator):
    def validate(self, document):
        ok = False
        try:
            weight = float(document.text)
        except ValueError:
            raise ValidationError(
                message='Please enter a number',
                cursor_position=len(document.text))  # Move cursor to end
        if 0 <= weight <= 150:
            ok = True
        if not ok:
            raise ValidationError(
                message='Shipment weight for air services cannot exceed 150 lbs',
                cursor_position=len(document.text))  # Move cursor to end


class FractionValidator(Validator):
    def validate(self, document):
        ok = False
        try:
            fraction = float(document.text)
        except ValueError:
            raise ValidationError(
                message='Please enter a number',
                cursor_position=len(document.text))  # Move cursor to end
        if 0 <= fraction <= 1:
            ok = True
        if not ok:
            raise ValidationError(
                message='Please enter a fraction between 0 and 1',
                cursor_position=len(document.text))  # Move cursor to end


class PercentileValidator(Validator):
    def validate(self, document):
        ok = False
        try:
            percentile = int(document.text)
        except ValueError:
            raise ValidationError(
                message='Please enter a number',
                cursor_position=len(document.text))  # Move cursor to end
        if 0 <= percentile <= 100:
            ok = True
        if not ok:
            raise ValidationError(
                message='Please enter a number between 0 and 100',
                cursor_position=len(document.text))  # Move cursor to end


models = []
for idx, model in enumerate(glob.glob(paths.model_dir + "/*.pkl.z")):
    models.append(model)

extracted_data_files = []
for idx, file_path in enumerate(glob.glob(paths.data_extracted_dir + "/*")):
    extracted_data_files.append(file_path)


main_menu_options = ['Run benchmarking',
                     'Run delivery prediction',
                     'Extract and preprocess shipment data from database',
                     'Calculate similarity matrix',
                     'Train delivery prediction model',
                     'Exit']

benchmarking_menu_options = ['Run benchmarking for another SID',
                            'Go back to main menu',
                            'Exit']

delivery_prediction_menu_options = ['Predict cost savings for one shipment',
                                    'Predict delivery time window probability for batch of shipments',
                                    'Go back to main menu']

predict_one_menu_options = ['Predict one shipment',
                            'Go back to main menu',
                            'Exit']

extract_menu_options = ['Extract data again',
                        'Go back to main menu',
                        'Exit']

benchmarking_metric_options = ["Avg spend",
                                'Avg spend per shipping method',
                                'Avg monthly spend',
                                'Avg discounts',
                                'Avg discounts per shipping method',
                                'Avg discounts per zone',
                                'Shipping method proportion',
                                'Shipper proportion',
                                'Volume shipped per shipping method']


benchmarking_initial_questions = [
    {
        'type': 'input',
        'name': 'sid',
        'message': 'Please enter the business SID that you wish to check on',
        'filter': lambda val: val.upper()
    },
    {
        'type': 'input',
        'name': 'percentile',
        'message': 'Please enter the similarity threshold (between 0 and 100)',
        'validate': PercentileValidator,
        'filter': lambda val: int(val)
    }
]

benchmarking_metrics_selection_questions = [
    {
        'type': 'checkbox',
        'qmark': '-',
        'message': 'Select benchmarking metrics',
        'name': 'benchmarking_metrics_selection',
        'choices': [
            # Separator('= Peers ='),
            # {
            #     'name': 'Number of peers',
            #     'checked': True
            # },
            Separator('= Spend ='),
            {
                'name': benchmarking_metric_options[0]
            },
            {
                'name': benchmarking_metric_options[1]
            },
            {
                'name': benchmarking_metric_options[2]
            },
            Separator('= Discounts ='),
            {
                'name': benchmarking_metric_options[3]
            },
            {
                'name': benchmarking_metric_options[4]
            },
            {
                'name': benchmarking_metric_options[5]
            },
            Separator('= Shipping ='),
            {
                'name': benchmarking_metric_options[6]
            },
            {
                'name': benchmarking_metric_options[7]
            },
            {
                'name': benchmarking_metric_options[8]
            }
        ],
        'validate': lambda answer: 'You must choose at least one metric.'
            if len(answer) == 0 else True
    }
]

delivery_prediction_one_questions = [
    {
        'type': 'input',
        'name': 'shipment_date',
        'message': 'What is the shipment date? (Please enter in YYYY-MM-DD format)',
        'validate': DateValidator,
    },
    {
        'type': 'input',
        'name': 'sender_zip',
        'message': 'What is the sender\'s zipcode?',
        'validate': ZipcodeValidator,
    },
    {
        'type': 'input',
        'name': 'recipient_zip',
        'message': 'What is the recipient\'s zipcode?',
        'validate': ZipcodeValidator,
    },
    {
        'type': 'input',
        'name': 'weight',
        'message': 'What is the shipment weight in lbs?',
        'validate': WeightValidator,
    },
    {
        'type': 'list',
        'name': 'shipper',
        'message': 'Which shipper?',
        'choices': ['Fedex', 'UPS'],
        'filter': lambda val: val.lower()
    }
]

extract_data_questions = [
    {
        'type': 'input',
        'name': 'start_date',
        'message': 'What is the start date for data extraction? (Please enter in YYYY-MM-DD format)',
        'validate': DateValidator,
        'default': '2018-01-01'
    },
    {
        'type': 'input',
        'name': 'end_date',
        'message': 'What is the cut-off date for data extraction? (Please enter in YYYY-MM-DD format)',
        'validate': DateValidator,
        'default': '2018-12-31'
    },
    {
        'type': 'input',
        'name': 'sample_size',
        'message': 'What is the fraction of data you wish to extract? (Please enter a fraction between 0 and 1)',
        'validate': FractionValidator,
        'default': '0.2'
    },

]


main_menu = [
    {
        'type': 'list',
        'name': 'main_menu',
        'message': '---Main Menu---',
        'choices': main_menu_options
    }
]

benchmarking_menu = [
    {
        'type': 'list',
        'name': 'benchmarking_menu',
        'message': '---Benchmarking---',
        'choices': benchmarking_menu_options
    }
]

delivery_prediction_menu = [
    {
        'type': 'list',
        'name': 'delivery_prediction_menu',
        'message': '---Delivery Prediction---',
        'choices': delivery_prediction_menu_options
    }
]

model_menu = [
    {
        'type': 'list',
        'name': 'model_menu',
        'message': 'Which model to use for prediction?',
        'choices': models
    }
]

predict_one_menu = [
    {
        'type': 'list',
        'name': 'predict_one_menu',
        'message': '---Predict cost savings for one shipment---',
        'choices': predict_one_menu_options
    }
]

extract_menu = [
    {
        'type': 'list',
        'name': 'extract_menu',
        'message': '---Extract data---',
        'choices': extract_menu_options
    }
]

