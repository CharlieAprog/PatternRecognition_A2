
# Setting up a virtual environment
Virtual environments will allow you to only have access to the requirements you have installed wile in the virtual environment. This prevents you from using a library that has not been mentioned in the requirements.txt. To create your own virtual environment follow the following steps
### Installing Virtual Environments
Venv is included with Python versions 3.3, if you have the newest version of python it should be included. Otherwise install via pip
> *pip install virtualenv*

Next, set up the virtual environment using the following command from the project directory
> *python3 -m venv venv*

Now you can activate and deactivate your venv using the following commands
#### Windows
>activate: *venv\Scripts\activate.bat*
>decativate: *deactivate*
#### Linux/Mac
>activate: *source venv/bin/activate*
>deactivate: *deactivate*

# Installing and saving the requirements
To install the requirements to make sure the code will run as expected,
>run: *pip install -r requirements.txt*

If new requirements have been added to the code AND YOU ARE USING A VENV, you can save the requirements by running the following command
>run: *pip freeze > requirements.txt*
