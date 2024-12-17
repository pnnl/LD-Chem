#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:38:52 2024

@author: fier887
"""

def get_number(string_val):
    if string_val.endswith('\n'):
        string_val = string_val[:-2]
        
    if '×' in string_val:
        idx = string_val.find('×')
        front_part = float(string_val[:idx])
        back_part = string_val[(idx+1):][2:]
        if back_part.startswith('−'):
            exponent = -float(back_part[1:])
        else:
            exponent = float(back_part)
        number = front_part*10.**exponent
    else:
        number = float(string_val)
    return number


