# Igor Kraszewski
def booth_function(x):
    funct = pow((x[0] + 2*x[1] - 7), 2) + pow((2*x[0] + x[1] - 5), 2)
    return funct


x = (-87.75386168033808, 58.33205762505378)
y = 15406.952509434128

print(booth_function(x), y)