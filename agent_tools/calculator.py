#!/usr/bin/env python3
"""
Secure mathematical computation for vibe coding RAG system 
Provides safe mathematical calculations without code execution risks
"""

import re
import math
from typing import Union

def secure_calculator(expression: str) -> str:
    """
    Perform secure mathematical calculations
    Only allows basic arithmetic operations and mathematical functions
    """
    try:
        # Input validation and sanitization
        if not expression or not isinstance(expression, str):
            return "Error: Invalid input"
        
        # Remove whitespace
        expression = expression.strip()
        
        # Check for maximum length to prevent DoS
        if len(expression) > 200:
            return "Error: Expression too long (max 200 characters)"
        
        # Allowed characters: digits, operators, parentheses, decimal points, math functions
        allowed_pattern = r'^[0-9+\-*/().\s\^sqrt\(\)log\(\)sin\(\)cos\(\)tan\(\)abs\(\)pow\(\),]*$'
        if not re.match(allowed_pattern, expression):
            return "Error: Invalid characters in expression"
        
        # Prevent dangerous patterns
        dangerous_patterns = [
            r'__.*__',  # Dunder methods
            r'import',  # Import statements
            r'exec',    # Code execution
            r'eval',    # Eval function
            r'open',    # File operations
            r'file',    # File operations
            r'input',   # Input functions
            r'raw_input',  # Input functions
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                return f"Error: Forbidden pattern detected: {pattern}"
        
        # Replace ^ with ** for Python exponentiation
        expression = expression.replace('^', '**')
        
        # Define safe mathematical functions
        safe_functions = {
            'sqrt': math.sqrt,
            'log': math.log,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'abs': abs,
            'pow': pow,
            'pi': math.pi,
            'e': math.e
        }
        
        # Create safe evaluation environment
        safe_dict = {
            '__builtins__': {},
            **safe_functions
        }
        
        # Evaluate expression safely
        try:
            result = eval(expression, safe_dict, {})
            
            # Check for valid numeric result
            if isinstance(result, (int, float, complex)):
                # Handle special cases
                if math.isnan(float(result)):
                    return "Error: Result is not a number (NaN)"
                elif math.isinf(float(result)):
                    return "Error: Result is infinite"
                else:
                    # Format result appropriately
                    if isinstance(result, float) and result.is_integer():
                        return str(int(result))
                    elif isinstance(result, float):
                        return f"{result:.10g}"  # Remove trailing zeros
                    else:
                        return str(result)
            else:
                return "Error: Result is not a valid number"
                
        except ZeroDivisionError:
            return "Error: Division by zero"
        except OverflowError:
            return "Error: Number too large"
        except ValueError as e:
            return f"Error: Invalid mathematical operation - {str(e)}"
        except Exception as e:
            return f"Error: Calculation failed - {str(e)}"
            
    except Exception as e:
        return f"Error: {str(e)}"

# Alias for backward compatibility
calculator = secure_calculator
