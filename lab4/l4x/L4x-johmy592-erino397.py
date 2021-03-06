# Run the script as follows:
# python L4x-johmy592-erino397.py < input_filename > output_filename

# This script gets the same results as the script given for the project. 

import sys


def trees(fp):
	buffer = []
	for line in fp:
		line = line.rstrip() 
		if not line.startswith('#'):
			if len(line) == 0:
				tree = [0]
				for w in buffer:
					tree.append(int(w[6]))
				yield tree, buffer
				buffer = []
			else:
				columns = line.split()
				if columns[0].isdigit(): 
					buffer.append(columns)


def make_A(tree):
	n = len(tree)
	A = [[float("inf") for i in range(n)] for j in range (n)]
	for i,head in enumerate(tree[1:],1):
		cnt = 0
		while head > 0:
			A[head][i] = cnt
			head = tree[head]
			cnt += 1
		A[0][i] = cnt
	return A


def make_T(tree, A):
	
	# Followed the same structure for eisner as described in the lecture.
	# Instead of finding maximum score, we find minimum cost where cost is defined as the
	# number of lifts required to transform the current tree to a corresponding projected tree
	n = len(tree)
	T1, T2, T3, T4 = ([[float("inf") if i!=j else 0 for i in range(n)] for j in range(n)] for _ in range(4))

	# Used as backpointers
	t1, t2, t3, t4 = ({} for _ in range(4))

	for k in range(1,n):
		for i in reversed(range(0, k)):
			# --- T4 ---
			best_j = None
			for j in range(i, k):
				temp_T4 = T2[i][j] + T1[j+1][k] + A[i][k]				
				if temp_T4 < T4[i][k]:
					T4[i][k] = temp_T4
					best_j = j
			if best_j is not None:
				t4[(i, k)] = t2.get((i, best_j), []) + t1.get((best_j + 1, k), []) + [(i, k)]
			
			# --- T3 ---
			best_j = None
			for j in range(i, k):
				temp_T3 = T2[i][j] + T1[j+1][k] + A[k][i]				
				if temp_T3 < T3[i][k]:
					T3[i][k] = temp_T3
					best_j = j
			if best_j is not None:
				t3[(i, k)] = t2.get((i, best_j), []) + t1.get((best_j + 1, k), []) + [(k, i)]

			# --- T2 ---
			best_j = None
			for j in range(i+1, k+1):
				temp_T2 = T4[i][j] + T2[j][k] 
				if temp_T2 < T2[i][k]:
					T2[i][k] = temp_T2
					best_j = j
			if best_j is not None:
				t2[(i, k)] = t4.get((i, best_j), []) + t2.get((best_j, k), [])
			
			# --- T1 ---
			best_j = None			
			for j in range(i, k):
				temp_T1 = T1[i][j] + T3[j][k]				
				if temp_T1 < T1[i][k]:
					T1[i][k] = temp_T1
					best_j = j
			if best_j is not None:
				t1[(i, k)] = t1.get((i, best_j), []) + t3.get((best_j, k), [])

	return T2[0][n - 1], t2[(0, n - 1)]


if __name__ == "__main__":
	tree = trees(sys.stdin)
	proj_trees = []
	for tree_list, buff in tree:
		cost, t2 = make_T(tree_list,make_A(tree_list))
		if cost != 0:
			for h in t2:
				tree_list[h[1]] = h[0]
		for w in buff:
			w[6] = str(tree_list[int(w[0])])
			print("\t".join(w))
		print()
		

