import sys
'''
    Run code with: python3 eisner.py my_rad_file.conllu
    Writes projective trees to my_rad_file.conllu_projective
'''

INF = float("inf")
tree = []
trees = []

# Reading trees from the given file
filename = sys.argv[1]
with open(filename) as f:
    for line in f:
        line = line.strip().split()
        if line:
            if line[0] == '#':
                continue
            idx = int(line[0])
            word = line[1]
            head = int(line[6])
            tree.append(line)
        else:
            trees.append(tree[:])
            tree = []

    trees.append(tree[:])

total = len(trees)-2
for tree_idx, tree in enumerate(trees):
    if not tree:
        continue
    #print("Progress: {0:.2f}".format(100*tree_idx/total), end="\r")
    n = len(tree)
    heads = [int(line[6]) for line in tree]

    # Score matrix
    A = [[INF for i in range(n+1)] for j in range(n+1)]
    for child in tree:
        idx = int(child[0])
        parent = int(child[6])
        cnt = 0
        while parent > 0:
            A[parent][idx] = cnt
            parent = heads[parent-1]
            cnt += 1
    A[0][idx] = cnt
    print(heads)
    print("A")
    for line in A:
        print (line)
    sys.exit(0)

    T1 = [[INF if i!=j else 0 for i in range(n+1)] for j in range(n+1)]
    T2 = [[INF if i!=j else 0 for i in range(n+1)] for j in range(n+1)]
    T3 = [[INF if i!=j else 0 for i in range(n+1)] for j in range(n+1)]
    T4 = [[INF if i!=j else 0 for i in range(n+1)] for j in range(n+1)]

    # Backtracking
    t1 = {}
    t2 = {}
    t3 = {}
    t4 = {}

    # Following the algorithm given in the hand out
    for m in range(1, n + 1):
        for s in range(n + 1):
            t = s+m
            if t > n:
                break

            best_q3 = -1
            best_q4 = -1
            for q in range(s,t):
                tmp3 = T2[s][q] + T1[q+1][t] + A[s][t]
                tmp4 = T2[s][q] + T1[q+1][t] + A[t][s]
                if tmp3 < T3[s][t]:
                    T3[s][t] = tmp3
                    best_q3 = q
                if tmp4 < T4[s][t]:
                    best_q4 = q
                    T4[s][t] = tmp4
            if best_q3 != -1:
                t3[(s,t)] = t2.get((s, best_q3), []) + t1.get((best_q3 + 1,t), []) + [(s,t)]
            if best_q4 != -1:
                t4[(s,t)] = t2.get((s, best_q4), []) + t1.get((best_q4 + 1, t), []) + [(t,s)]

            best_q1 = -1
            best_q2 = -1
            for q in range(s,t):
                tmp1 = T4[s][q + 1] + T1[q + 1][t]
                tmp2 = T2[s][q] + T3[q][t]
                if tmp1 < T1[s][t]:
                    T1[s][t] = tmp1
                    best_q1 = q + 1
                if tmp2 < T2[s][t]:
                    T2[s][t] = tmp2
                    best_q2 = q
            if best_q1 != -1:
                t1[(s,t)] = t4.get((s, best_q1), []) + t1.get((best_q1,t), [])
            if best_q2 != -1:
                t2[(s,t)] = t2.get((s, best_q2), []) + t3.get((best_q2,t), [])

    # The total cost for lifting
    if T2[0][n] != 0 and T2[0][n] != float("inf"):     
        print("Cost:", T2[0][n]) 
    #sys.exit(0)
