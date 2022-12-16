# shop/management/commands/populate_db.py

import random
from datetime import datetime, timedelta
from secrets import choice

import pytz
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model

from apps.home.models import Project, Expense, Income
from apps.authentication.models import User


class Command(BaseCommand):
    """To launch, type 'python manage.py populate_db --amount 1000' in the command line"""
    help = 'Populates the database with random generated data.'

    def add_arguments(self, parser):
        parser.add_argument('--amount', type=int, help='The number of purchases that should be created.')

    def handle(self, *args, **options):
        dt = pytz.utc.localize(datetime.now() - timedelta(days=random.randint(0, 1825)))
        user= User.objects.get(id=1)
        amount = options['amount'] if options['amount'] else 2500
        job= Project.objects.filter(user=user)
        project_choice=list(job)
        for i in range(0, amount):
            dt = pytz.utc.localize(datetime.now() - timedelta(days=random.randint(0, 1825)))
            expense = Expense.objects.create(
                category =random.choice(Expense.EXPENSE_CHOICES)[0],
                amount = random.randint(200, 31500),
                date= dt,
                project_name= random.choice(project_choice),
                user=user,
            )            
            income = Income.objects.create(
                category =random.choice(Income.INCOME_CHOICES)[0],
                amount = random.randint(200, 31500),
                date= dt,
                project_name= random.choice(project_choice),
                user=user,
                )
        self.stdout.write(self.style.SUCCESS('Successfully populated the database.'))
