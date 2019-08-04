from __future__ import print_function, unicode_literals
import datetime
import glob
import zipcodes
from questionary import Separator,Validator, ValidationError
import paths
from prompt_toolkit.styles import Style

blue = Style([
    ('separator', 'fg:#6C6C6C'),
    ('qmark', 'fg:#E91E63 bold'),
    ('question', ''),
    ('selected', 'fg:#2196F3 bold'),
    ('pointer', 'fg:#2196F3 bold'),
    ('answer', 'fg:#2196F3 bold'),
    # Token.QuestionMark: '#E91E63 bold',
    # Token.Selected: '#673AB7 bold',
    # Token.Instruction: '',  # default
    # Token.Answer: '#2196f3 bold',
    # Token.Question: '',
])
custom_style_fancy = Style([
    ('separator', 'fg:#cc5454'),
    ('qmark', 'fg:#673ab7 bold'),
    ('question', ''),
    ('selected', 'fg:#cc5454'),
    ('pointer', 'fg:#673ab7 bold'),
    ('answer', 'fg:#f44336 bold'),
])

custom_style_dope = Style([
    ('separator', 'fg:#6C6C6C'),
    ('qmark', 'fg:#FF9D00 bold'),
    ('question', ''),
    ('selected', 'fg:#5F819D'),
    ('pointer', 'fg:#FF9D00 bold'),
    ('answer', 'fg:#5F819D bold'),
])

custom_style_genius = Style([
    ('qmark', 'fg:#E91E63 bold'),
    ('question', ''),
    ('selected', 'fg:#673AB7 bold'),
    ('answer', 'fg:#2196f3 bold'),
])

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

benchmarking_preprocess_initial_options = ['Average of all',
                                   'Volumetric scale',
                                   'Volumetric proportion by zone',
                                   'Volumetric proportion by month',
                                   'Weight scale',
                                   'Weight scale by zone',
                                   'Weight scale by month']

benchmarking_preprocess_menu_options = ['Create another similarity matrix',
                                        'Go back to main menu',
                                        'Exit']

delivery_prediction_preprocess_menu_options = ['Train a new model',
                                                'Go back to main menu',
                                                'Exit']

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


class YearMonthValidator(Validator):
    '''
    abc
    '''
    def validate(self, document):
        try:
            datetime.datetime.strptime(document.text, '%Y-%m')
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


class SIDValidator(Validator):
    def validate(self, document):
        import builtins
        ok = False
        if document.text.upper() in builtins.sid_list:
            ok = True
        if not ok:
            raise ValidationError(
                message='SID not found in selected similarity matrix.',
                cursor_position=len(document.text))  # Move cursor to end


class sumToOneValidator(Validator):
    def validate(self, document):
        import builtins
        ok = False
        if document.text.upper() in builtins.sid_list:
            ok = True
        if not ok:
            raise ValidationError(
                message='SID not found in selected similarity matrix.',
                cursor_position=len(document.text))  # Move cursor to end


models = []
for idx, model in enumerate(glob.glob(paths.model_dir + "/*.pkl.z")):
    models.append(model)

extracted_data_files = []
for idx, file_path in enumerate(glob.glob(paths.data_extracted_dir + "/*")):
    extracted_data_files.append(file_path)

similarity_matrices = []
for idx, file_path in enumerate(glob.glob(paths.data_benchmarking_dir + "/*similarity*")):
    similarity_matrices.append(file_path)

KPI_databases = []
for idx, file_path in enumerate(glob.glob(paths.data_benchmarking_dir + "/*KPI*")):
    KPI_databases.append(file_path)

benchmarking_matrix_kpidatabase = [
    {
        'type': 'list',
        'name': 'similarity_matrix',
        'message': 'Which similarity matrix to use for calculation?',
        'choices': similarity_matrices
    },
    {
        'type': 'list',
        'name': 'kpi_database',
        'message': 'Which KPI database to use for calculation?',
        'choices': KPI_databases
    },
]

benchmarking_kpi = [
    {
        'type': 'input',
        'name': 'sid',
        'message': 'Please enter the business SID that you wish to check on',
        'validate': SIDValidator,
        'filter': lambda val: val.upper()
    },
    {
        'type': 'input',
        'name': 'percentile',
        'message': 'Please enter the similarity threshold (between 0 and 100)',
        'validate': PercentileValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'checkbox',
        'qmark': '-',
        'message': 'Select benchmarking metrics',
        'name': 'kpi_selected',
        'choices': [
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
        'message': 'What is the start year/month for data extraction? (Please enter in YYYY-MM format)',
        'validate': YearMonthValidator,
        'default': '2018-06'
    },
    {
        'type': 'input',
        'name': 'end_date',
        'message': 'What is the cut-off year/month for data extraction? (Please enter in YYYY-MM format)',
        'validate': YearMonthValidator,
        'default': '2019-05'
    },
    {
        'type': 'input',
        'name': 'sample_size',
        'message': 'What is the fraction of data you wish to extract? (Please enter a fraction between 0 and 1)',
        'validate': FractionValidator,
        'default': '0.2'
    },

]

delivery_prediction_menu = [
    {
        'type': 'list',
        'name': 'delivery_prediction_menu',
        'message': '---Delivery Prediction---',
        'choices': delivery_prediction_menu_options
    }
]

delivery_prediction_model = [
    {
        'type': 'list',
        'name': 'delivery_prediction_model_choice',
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

benchmarking_metric_weights = [
    {
        'type': 'input',
        'name': 'weight_vs',
        'message': 'Volumetric Scale (VS) Weight',
        'validate': FractionValidator,
        'default': '0.1',
        'filter': lambda val: float(val)
    },
    {
        'type': 'input',
        'name': 'weight_vpz',
        'message': 'Volumetric Proportion by Zone (VPZ) Weight',
        'validate': FractionValidator,
        'default': '0.1',
        'filter': lambda val: float(val)
    },
    {
        'type': 'input',
        'name': 'weight_vpm',
        'message': 'Volumetric Proportion by Month (VPM) Weight',
        'validate': FractionValidator,
        'default': '0.1',
        'filter': lambda val: float(val)
    },
    {
        'type': 'input',
        'name': 'weight_ws',
        'message': 'Weight Scale (WS) Weight',
        'validate': FractionValidator,
        'default': '0.1',
        'filter': lambda val: float(val)
    },
    {
        'type': 'input',
        'name': 'weight_wpz',
        'message': 'Weight Proportion by Zone (WPS) Weight',
        'validate': FractionValidator,
        'default': '0.1',
        'filter': lambda val: float(val)
    },
    {
        'type': 'input',
        'name': 'weight_wpm',
        'message': 'Weight Proportion by Month (WPM) Weight',
        'validate': FractionValidator,
        'default': '0.1',
        'filter': lambda val: float(val)
    }
]



benchmarking_preprocess_questions = [
    {
        'type': 'list',
        'name': 'benchmarking_preprocess_answer',
        'message': 'How do you want to generate the new similarity matrix?',
        'choices': benchmarking_preprocess_initial_options
    }
]

benchmarking_preprocess_menu = [
    {
        'type': 'list',
        'name': 'benchmarking_preprocess_menu',
        'message': '---Benchmarking: Create similarity matrix ---',
        'choices': benchmarking_preprocess_menu_options
    }
]

extracted_data = [
    {
        'type': 'list',
        'name': 'extracted_data_choice',
        'message': 'Select extracted data file to train a new model on',
        'choices': extracted_data_files
    }
]

delivery_prediction_preprocess_menu = [
    {
        'type': 'list',
        'name': 'delivery_prediction_preprocess_menu',
        'message': '---Delivery Prediction: Train a new model ---',
        'choices': delivery_prediction_preprocess_menu_options
    }
]