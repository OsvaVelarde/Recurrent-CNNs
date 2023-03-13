'''
Title: Functions - Parser for arguments.
Author: Osvaldo M Velarde
Project: Feedback connections in visual system
'''

# ===========================================================
def parse_coords(value):
    x, y = map(int, value.split(','))
    return x, y

def parse_boolean(value):
    value = value.lower()
    if value == 'true':
    	return True
    else:
    	return False

def parse_none(value):
    value = value.lower()
    if value == 'none':
    	return None
    else:
    	return value
# ===========================================================