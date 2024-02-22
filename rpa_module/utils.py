
def format_value(value, line_width):
    """
    Format normalized reaction rate value
    
    This converts a reaction flux to percentage
    and format it into a string to label a graph
    edge according to its value
    """
    # Convert to %
    value *= (100/line_width)
    # No decimals
    if value > 1.0:
        label = f"{value:.0f}%"
    # One decimal
    elif value > 0.1:
        label = f"{value:.2f}%"
    # Two decimal
    elif value > 0.01:
        label = f"{value:.3f}%"
    # Scientific notation with no
    # decimals
    else:
        label = f"{value:.0e}%"
    return label


