# --coding=utf-8--
from scipy.special import comb, perm

# perm(3,2)=6
# comb(3,2)=3

d = 3*1024*700  # img size d
k = 45  # ablation parameter k
n1 = 3*20*20
n2 = 3*40*40
n3 = 3*81*81
n4 = 3*163*163

print("Size 20: Upper bound possibility: C_{d-n}^k/C_d^k \n", comb(d-n1, k)/comb(d,k))
print("Size 20: Lower bound possibility: C_n^k/C_d^k \n", comb(n1, k)/comb(d,k))

print("Size 40: Upper bound possibility: C_{d-n}^k/C_d^k \n", comb(d-n2, k)/comb(d,k))
print("Size 40: Lower bound possibility: C_n^k/C_d^k \n", comb(n2, k)/comb(d,k))

print("Size 81: Upper bound possibility: C_{d-n}^k/C_d^k \n", comb(d-n3, k)/comb(d,k))
print("Size 81: Lower bound possibility: C_n^k/C_d^k \n", comb(n3, k)/comb(d,k))

print("Size 163: Upper bound possibility: C_{d-n}^k/C_d^k \n", comb(d-n4, k)/comb(d,k))
print("Size 163: Lower bound possibility: C_n^k/C_d^k \n", comb(n4, k)/comb(d,k))
