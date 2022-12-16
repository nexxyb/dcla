# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.messages.views import SuccessMessageMixin
#from apps.home.models import  Project
from .forms import LoginForm, SignUpForm, User
from django.views.generic.edit import UpdateView, CreateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import generic
from django.urls import reverse, reverse_lazy
from random import randint
from django.utils.decorators import method_decorator
from apps.home.decorators import apm_required


def login_view(request):
    form = LoginForm(request.POST or None)

    msg = None

    if request.method == "POST":

        if form.is_valid():
            email = form.cleaned_data.get("email")
            password = form.cleaned_data.get("password")
            user = authenticate(email=email, password=password)
            if user is not None:
                login(request, user)
                if user.is_apm:
                    return redirect("/")
                elif user.is_pm:
                    return redirect("dashboard")
            else:
                msg = 'Invalid credentials'
        else:
            msg = 'Error validating the form'

    return render(request, "accounts/login.html", {"form": form, "msg": msg})


def register_user(request):
    msg = None
    success = False

    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            email = form.cleaned_data.get("email")
            raw_password = form.cleaned_data.get("password1")
            user = authenticate(email=email, password=raw_password)

            msg = 'User created successfully.'
            success = True

            # return redirect("/login/")

        else:
            msg = 'Form is not valid'
    else:
        form = SignUpForm()

    return render(request, "accounts/register.html", {"form": form, "msg": msg, "success": success})
    
class Profile(generic.DetailView, LoginRequiredMixin):
    model = User
    template_name = 'accounts/profile.html'
    
    def get_context_data(self, **kwargs):
         # Call the base implementation first to get a context
        context=super(Profile, self).get_context_data(**kwargs)
        context['project_list']=Project.objects.filter(user=self.request.user)
        return context

class UpdateProfileView(UpdateView, LoginRequiredMixin):
    model= User
    fields= ['first_name', 'last_name', 'bio', 'location']
    template_name = 'accounts/company_form.html'
    success_url = "profile"
    
    def get_success_url(self):
        slug = self.kwargs["slug"]
        return reverse("profile", kwargs={"slug": slug})
    
@method_decorator(apm_required, name='dispatch')
class AccountUpgradeView(LoginRequiredMixin, UpdateView):
    model= User
    fields= ['company_name']
    template_name= 'accounts/account_upgrade_form.html'
    
    def form_valid(self, form):
        form.instance.is_pm = True
        form.instance.is_apm = False
        return super().form_valid(form)