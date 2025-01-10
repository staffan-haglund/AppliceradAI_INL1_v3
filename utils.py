import os

def new_screen():
    '''
    Clears the screen, for Windows and Unix
    '''
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')