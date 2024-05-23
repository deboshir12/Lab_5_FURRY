def f(n):
    if n > 1:
        return n + f(n - 1)
    else:
        return 1
    
print(f(5))

# N,M = list(map(int, m.readline().split()))
# f = [list(map(int, x.split(' '))) for x in m]
# for i in range(int(N)):
#   f.append([0, 0])
# print(f)
# for i in range(len(M)):
#   a = list(input().split(' '))
f = [[1,2], [3,4], [2,1]]
print(sorted(f, key=lambda x: x[0]))