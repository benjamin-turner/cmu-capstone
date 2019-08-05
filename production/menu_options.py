"""
Static menu options and styles

This script contains the static style attributes, user input validators and static menu options for the UI.
Any questions contained in this module should be imported with prompt().

References
    Questionary: https://github.com/tmbo/questionary

"""
from __future__ import print_function, unicode_literals
import datetime
import zipcodes
from prompt_toolkit.styles import Style
from questionary import Separator, Validator, ValidationError
from dateutil import relativedelta

blue = Style([
    ('separator', 'fg:#6C6C6C'),
    ('qmark', 'fg:#E91E63 bold'),
    ('question', ''),
    ('selected', 'fg:#2196F3 bold'),
    ('pointer', 'fg:#2196F3 bold'),
    ('answer', 'fg:#2196F3 bold'),
])

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

class DateValidator(Validator):
    """Contains class methods to validate date strings

    """
    def validate(self, document):
        """Validates date strings

        Args:
            document: User input to questionary prompt

        Raises:
            ValidationError if date is not in valid format

        Returns:
            True if date is in valid format
        """
        try:
            datetime.datetime.strptime(document.text, '%Y-%m-%d')
        except ValueError:
            raise ValidationError(
                message='Please enter a valid date in YYYY-MM-DD format',
                cursor_position=len(document.text))  # Move cursor to end


class YearMonthValidator(Validator):
    """Contains class method to validate year and month

    """
    def validate(self, document):
        """Validates year month strings

        Args:
            document: User input to questionary prompt

        Raises:
            ValidationError if year and month is not in valid format

        Returns:
            True if date is in valid format
        """
        try:
            datetime.datetime.strptime(document.text, '%Y-%m')
        except ValueError:
            raise ValidationError(
                message='Please enter a valid date in YYYY-MM-DD format',
                cursor_position=len(document.text))  # Move cursor to end


class EndYearMonthValidator(Validator):
    """Contains class method to validate end range of year and month

    """

    def validate(self, document):
        """Validates year month strings

        Args:
            document: User input to questionary prompt

        Raises:
            ValidationError if year and month is not in valid format OR range between start month and end month
                is less than 12 months.

        Returns:
            True if date is in valid format
        """
        ok = False
        try:
            # datetime.datetime.strptime(document.text, '%Y-%m')
            import builtins
            start_year_month = builtins.start_year_month
            start_date = datetime.datetime.strptime(start_year_month + "-1", "%Y-%m-%d").date()
            end_date = datetime.datetime.strptime(document.text + "-1", "%Y-%m-%d").date()
        except ValueError:
            raise ValidationError(
                message='Please enter a valid date in YYYY-MM-DD format',
                cursor_position=len(document.text))  # Move cursor to end
        delta = relativedelta.relativedelta(end_date, start_date)
        num_months = delta.years * 12 + delta.months + 1
        if num_months >= 12:
            ok = True
        if not ok:
            raise ValidationError(
                message='The time period for data extraction must be >= 12 months',
                cursor_position=len(document.text))  # Move cursor to end

class ZipcodeValidator(Validator):
    """Contains class method to validate zipcodes

    """
    def validate(self, document):
        """Validates zipcodes

        Args:
            document: User input to questionary prompt

        References:
            Zipcodes: https://pypi.org/project/zipcodes/

        Returns:
            True if user zipcode is valid number with len=5 and according to zipcodes package

        """
        ok = False
        if document.text.isdigit() and len(document.text) == 5:
            ok = zipcodes.is_real(document.text)
        if not ok:
            raise ValidationError(
                message='Please enter a valid zipcode',
                cursor_position=len(document.text))  # Move cursor to end


class NumberValidator(Validator):
    """Contains class method to validate numbers

    """
    def validate(self, document):
        """Validates number

        Args:
            document: User input to questionary prompt.

        Raises:
            ValidationError if number is not valid.

        Returns:
            True if number is valid.
        """
        try:
            float(document.text)
        except ValueError:
            raise ValidationError(
                message='Please enter a number',
                cursor_position=len(document.text))  # Move cursor to end


class WeightValidator(Validator):
    """Contains class method to validate weight of a shipment

    """
    def validate(self, document):
        """Validates shipment weight

        Args:
            document: User input to questionary prompt

        Raises:
            ValidationError if weight is not valid

        Returns:
            True if weight is a number and below 150 lbs. According to FedEx and UPS,
            air shipment weight cannot exceed 150 lbs.
        """
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
    """Contains class method to validate fractions

    """
    def validate(self, document):
        """Validates fractions.

        Args:
            document: User input to questionary prompt

        Raises:
            ValidationError if fraction is not valid

        Returns:
            True if fraction is valid.
        """
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
    """Contains class method to validate percentile

    """
    def validate(self, document):
        """Validates percentiles

        Args:
            document: User input to questionary prompt

        Raises:
            ValidationError if percentile is not valid

        Returns:
            True if percentile is a number and between 0 and 100.
        """
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
    """Contains class method to validate business SID

    """
    def validate(self, document):
        """Validates business SID against similarity matrix provided.

        Args:
            document: User input to questionary prompt

        Raises:
            ValidationError if business SID is not found.

        Returns:
            True if SID is found in columns of similarity matrix provided.
        """
        import builtins
        ok = False
        if document.text.upper() in builtins.sid_list:
            ok = True
        if not ok:
            raise ValidationError(
                message='SID not found in selected similarity matrix.',
                cursor_position=len(document.text))  # Move cursor to end

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
        'default': '2019-07-04',
        'validate': DateValidator,
    },
    {
        'type': 'input',
        'name': 'sender_zip',
        'message': 'What is the sender\'s zipcode?',
        'default': '15213',
        'validate': ZipcodeValidator,
    },
    {
        'type': 'input',
        'name': 'recipient_zip',
        'message': 'What is the recipient\'s zipcode?',
        'default': '33324',
        'validate': ZipcodeValidator,
    },
    {
        'type': 'input',
        'name': 'weight',
        'message': 'What is the shipment weight in lbs?',
        'default': '10',
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



