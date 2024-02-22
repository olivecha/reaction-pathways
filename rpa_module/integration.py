"""
Integration Functions
"""


def simpson_13(fx, x):
    """
    Simpson's 1/3 integration for a 3-point irregular grid

    Parameters:
    - x: grid points {array of len(3)}
    - fx: function evaluated at each grid point

    Returns:
    - Integral estimate using Simpson's 1/3 rule.


    Reference: P. Versailles, CH formation in Premixed Flames of C1–C4 Alkanes: 
               Assessment of Current Chemical Modelling Capability Against 
               Experiments, McGill University, 2017 Ph.D. thesis. 
               http://digitool.library.mcgill.ca/R/
    """
    if (len(x) != 3 or len(fx) != 3):
        raise ValueError("x and fx must be 1x3 or 3x1 matrices")
    
    
    fd_x0x1 = (fx[1] - fx[0]) / (x[1] - x[0])
    fd_x1x2 = (fx[2] - fx[1]) / (x[2] - x[1])
    fd_x0x1x2 = (fd_x1x2 - fd_x0x1) / (x[2] - x[0])
    
    A = fx[0] - fd_x0x1 * x[0] + fd_x0x1x2 * x[0] * x[1]
    B = fd_x0x1 - (x[0] + x[1]) * fd_x0x1x2
    C = fd_x0x1x2
    
    Int = A * (x[2] - x[0]) + B * (x[2] ** 2 - x[0] ** 2) / 2 + C * (x[2] ** 3 - x[0] ** 3) / 3
    
    return Int


def simpson_13_comp(fx, x):
    """
    Composite Simpson's 1/3 integration on an irregular grid 
    of arbitrary length

    Parameters:
    - x: grid points
    - fx: function evaluated at each grid point

    Returns:
    - Integral estimate using composite Simpson's 1/3 rule.

    Reference: P. Versailles, CH formation in Premixed Flames of C1–C4 Alkanes: 
               Assessment of Current Chemical Modelling Capability Against 
               Experiments, McGill University, 2017 Ph.D. thesis. 
               http://digitool.library.mcgill.ca/R/
    """

    
    if len(x) != len(fx):
        raise ValueError("x and fx must be of the same size")
    
    # Remove duplicate x-values
    unq_x = []
    unq_fx = []
    
    for i, val in enumerate(x):
        if val not in unq_x:
            unq_x.append(val)
            unq_fx.append(fx[i])

    x = unq_x
    fx = unq_fx
 
      
    if len(x) <= 3 and len(fx) <= 3:
        raise ValueError("x and fx must be 1x>3 or >3x1 matrices")
    
    nb_gp = len(x)
    nb_Intint = (nb_gp - 1) // 2

    Int = 0
    
    # Apply Simpson's 1/3 rule to each interval
    for i in range(nb_Intint):
        Int += simpson_13(fx[2*i:2*i+3], x[2*i:2*i+3])
        
    
    # Handle the last interval if the number of 3-gridpoint intervals is not a round number
    if 2 * nb_Intint + 1 < nb_gp:
        Int += (fx[-1] + fx[-2]) / 2 * (x[-1] - x[-2])
    
    return Int
