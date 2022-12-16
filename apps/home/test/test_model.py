from django.test import TestCase

from apps.home.models import Expense, Income, Project
from apps.authentication.models import User

class ProjectModelTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        user1= User.objects.create(email = 'nexxy@example.com')
        project=Project.objects.create(
                    project_name= 'Atlanta Heights',
                    project_amount= "40000",
                    start_date= '2020-6-2',
                    end_date= '2022-6-23',                    
                    user=user1
                )
        expense = Expense.objects.create(
                category ='grocery',
                amount = 1200,
                date= '2022-6-3',
                project_name= project,
                user=user1,
            )            
        income = Income.objects.create(
                category = 'commission',
                amount = 1300,
                date= '2022-6-3',
                project_name= project,
                user=user1,
                )
    def test_project_duration(self):
        """Test that Project end date is after start date"""
        project= Project.objects.get(project_name="Atlanta Heights")
        self.assertTrue(project.start_date < project.end_date)
        
    def test_project_amount_currency(self):
        """Test that project amount currency is as saved"""
        project= Project.objects.get(project_name="Atlanta Heights")
        self.assertContains(project.project_amount, '$')
        
    def test_total_expenses(self):
        """Test total expense made  """
        project= Project.objects.get(project_name="Atlanta Heights")
        self.assertEqual(project.total_spent(), '$1,200')
        
    def test_total_income(self):
        """Test total income received  """
        project= Project.objects.get(project_name="Atlanta Heights")
        self.assertEqual(project.total_income(), '$1,300')
        
    def test_budget_balance(self):
        """Test project budget balance  """
        project= Project.objects.get(project_name="Atlanta Heights")
        self.assertEqual(project.budget_balance(), '$38,800')
    
    def test_actual_balance(self):
        """Test project balance   """
        project= Project.objects.get(project_name="Atlanta Heights")
        self.assertEqual(project.actual_balance(), '$100')

    
class ExpenseModelTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        user1= User.objects.create(email = 'nexxy@example.com')
        project=Project.objects.create(
                    project_name= 'Atlanta Heights',
                    project_amount= "40000",
                    start_date= 2020-6-2,
                    end_date= 2022-6-23,
                    company_code=23589,
                    user=user1
                )
        expense = Expense.objects.create(
                category ='grocery',
                amount = 1200,
                date= 2022-6-3,
                project_name= project,
                user=user1,
            )            
        income = Income.objects.create(
                category = 'commission',
                amount = 1300,
                date= 2022-6-3,
                project_name= project,
                user=user1,
                )
        
