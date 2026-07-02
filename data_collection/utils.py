def human_readable_number(num):
    if num < 1e6:
        return f"{int(num/1e3)}K"
    elif num < 1e9:
        return f"{num/1e6:.1f}M"
    elif num < 1e12:
        return f"{num/1e9:.1f}B"
    else:
        return str(num)