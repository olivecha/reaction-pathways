
def format_value(value, max_width):
    """
    Format normalized reaction rate value
    
    This converts a reaction flux to percentage
    and format it into a string to label a graph
    edge according to its value
    """
    # This would not work
    if value > max_width:
        raise ValueError(f"edge value {value}  is more"
                          " than the maximum value used"
                          " ({max_width}), so normalization"
                          " wont work here as the"
                          " percentage will be > 100%")
    # Normalize and convert to %
    value *= (100/max_width)
    # No decimals
    if value > 10:
        label = f"{value:.0f}%"
    # One decimal
    elif value > 1:
        label = f"{value:.1f}%"
    # Two decimal
    elif value > 0.1:
        label = f"{value:.2f}%"
    # Scientific notation with no decimals
    else:
        label = f"{value:.1e}%"
    return label


