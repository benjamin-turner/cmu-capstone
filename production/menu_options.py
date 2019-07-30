from __future__ import print_function, unicode_literals
import datetime
import glob
import sys
import joblib
import zipcodes
import regex
import os
from pprint import pprint

from PyInquirer import style_from_dict, Token, prompt
from PyInquirer import Validator, ValidationError

import delivery_prediction_predict
import paths

from examples import custom_style_2

class DateValidator(Validator):
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

models = []
for idx, model in enumerate(glob.glob(paths.model_dir + "/*.pkl.z")):
    models.append(model)

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
        'validate': NumberValidator,
    },
    {
        'type': 'list',
        'name': 'shipper',
        'message': 'Which shipper?',
        'choices': ['Fedex', 'UPS'],
        'filter': lambda val: val.lower()
    }
]

delivery_prediction_menu_options = ['Predict cost savings for one shipment',
                                    'Predict delivery time window probability for batch of shipments',
                                    'Go back to main menu']

main_menu_options = ['Run benchmarking',
                     'Run delivery prediction',
                     'Extract shipment data from database',
                     'Calculate similarity matrix',
                     'Train delivery prediction model',
                     'Exit']

predict_one_menu_options = ['Predict one shipment',
                            'Go back to main menu',
                            'Exit']

main_menu = [
    {
        'type': 'list',
        'name': 'main_menu',
        'message': '---Main Menu---',
        'choices': main_menu_options
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


