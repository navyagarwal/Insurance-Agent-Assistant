from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField, SelectField, BooleanField

class profileForm(FlaskForm):
    age = IntegerField('age')
    gender = SelectField('gender', choices=[(0, 'Male'), (1, 'Female')])
    income = IntegerField('income')
    residence = SelectField('residence', choices=[(1, 'Metro'), (0, 'Non-metro')])
    diabetes = BooleanField('diabetes')
    heart = BooleanField('heart')
    bp = BooleanField('bp')
    other = BooleanField('other')
    surg = BooleanField('surgical procedure')
    covid = BooleanField('covid')
    submit = SubmitField('Sign Up')