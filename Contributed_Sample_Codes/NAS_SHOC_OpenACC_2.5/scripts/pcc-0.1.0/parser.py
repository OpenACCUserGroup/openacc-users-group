'''
Created on Jul, 2016

@author: pinogal
'''

from ply import *

import scanner
import sys

tokens = scanner.tokens
global res_parser

def printNode(n, space):
    print "-"*space + str(n)

#------
# Rules
#_-----

def p_Program(p):
    '''
    Program :    StmtList
    '''

def p_StmtList(p):
    '''
    StmtList :    StmtList Stmt
            |    Stmt
    '''

def p_Stmt(p):
    '''
    Stmt :    ExecTime
        |    Class
        |    Verification
        |    AccelTiming
        |    error
    '''

def p_ExecTime(p):
    '''
    ExecTime :    TIME IN SECONDS EQUALS FCONST
    '''
    print ("ExecTime(s): " + str(p[5]))

def p_Class(p):
    '''
    Class :    CLASS EQUALS ID
    '''
    print ("Class: " + str(p[3]))

def p_Verification(p):
    '''
    Verification :    VERIFICATION EQUALS ID
    '''
    print ("Verification: " + str(True if p[3].lower() == "successful" else False))
    
def p_AccelTiming(p):
    '''
    AccelTiming :    ID NVIDIA DEVICENUM EQUALS ICONST TIME LPAREN ID RPAREN COLON Timing
    '''
    print "AccelTiming<{} NVIDIA DEVICENUM={}>(s) : {}".format(p[1], p[5], float(p[11])/1.0e6)

def p_Timing(p):
    '''
    Timing :    Timing COMMA ICONST
            |    ICONST
    '''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = p[1] + p[3]

# Error rule for syntax errors
# Catastrophic error handler
def p_error(p):
    # Read ahead looking for a terminating ";"
    pass
#     res_parser.restart()
#     if p:
#         line = "%s = %r at line %d near %s " % (p.type, p.value, p.lineno, p.value)
#         while True:
#             tok = res_parser.token()             # Get the next token
#             if not tok or tok.type == 'SEMI': 
#                 break
#             else:
#                 line = line + tok.value
     
#         if len(line) > 0:
#             print("Syntax error at token " + line)
#     else:
#         print("Syntax error at EOF")

# Build the parser
res_parser = yacc.yacc()

def parse(data, debug_flag=False):
    res_parser.error = 0
    prog = res_parser.parse(data, debug=debug_flag)
    if res_parser.error: return None
    return prog

if __name__ == '__main__':
    if len(sys.argv) == 2:
        data = open(sys.argv[1]).read()
        prog = parse(data, False)