# defining a function to convert 
# integer part to binary
def binarycode(m):
      
    a = []
    n = int(m)
      
    while n != 0:
          
        a.append(int(n % 2))
        n = n//2
          
    a.reverse()
    return a
  
# Driver Code
print(binarycode(0.58663307*(10**100)))