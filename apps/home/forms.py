from django import forms

class DataCleaningForm(forms.Form):
    remove_nulls = forms.BooleanField(required=False, label='Remove Null Values')
    remove_duplicates = forms.BooleanField(required=False, label='Remove Duplicate Rows')
    remove_outliers = forms.BooleanField(required=False, label='Remove Outliers')
    standardize_columns = forms.BooleanField(required=False, label='Standardize Columns')

    remove_nulls_columns = forms.CharField(required=False, widget=forms.TextInput(attrs={'class': 'special'}))
    remove_duplicates_columns = forms.CharField(required=False, widget=forms.TextInput(attrs={'class': 'special'}))
    remove_outliers_columns = forms.CharField(required=False, widget=forms.TextInput(attrs={'class': 'special'}))
    standardize_columns_columns = forms.CharField(required=False, widget=forms.TextInput(attrs={'class': 'special'}))

    # Add hidden fields to include the values of the text input fields in the POST data
    remove_nulls_options = forms.CharField(required=False, widget=forms.HiddenInput())
    remove_duplicates_options = forms.CharField(required=False, widget=forms.HiddenInput())
    remove_outliers_options = forms.CharField(required=False, widget=forms.HiddenInput())
    standardize_columns_options = forms.CharField(required=False, widget=forms.HiddenInput())
