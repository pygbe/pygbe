"""
Analytical solution of spherical molecule in ionic solution
From Kirkwood 1934
"""
from numpy import *
from scipy import special
from scipy.special._cephes import lpmv
from scipy.misc import factorial

def an_spherical(q, xq, E_1, E_2, E_0, R, N):
        
    PHI = zeros(len(q))
    for K in range(len(q)):
        rho = sqrt(sum(xq[K]**2))
        zenit = arccos(xq[K,2]/rho)
        azim  = arctan2(xq[K,1],xq[K,0])

        phi = 0.+0.*1j
        for n in range(N):
            for m in range(-n,n+1):
                sph1 = special.sph_harm(m,n,zenit,azim)
                cons1 = rho**n/(E_1*E_0*R**(2*n+1))*(E_1-E_2)*(n+1)/(E_1*n+E_2*(n+1))
                cons2 = 4*pi/(2*n+1)

                for k in range(len(q)):
                    rho_k   = sqrt(sum(xq[k]**2))
                    zenit_k = arccos(xq[k,2]/rho_k)
                    azim_k  = arctan2(xq[k,1],xq[k,0])
                    sph2 = conj(special.sph_harm(m,n,zenit_k,azim_k))
                    phi += cons1*cons2*q[K]*rho_k**n*sph1*sph2
        
        PHI[K] = real(phi)/(4*pi)

    return PHI

def get_K(x,n):

    K = 0.
    n_fact = factorial(n)
    n_fact2 = factorial(2*n)
    for s in range(n+1):
        K += 2**s*n_fact*factorial(2*n-s)/(factorial(s)*n_fact2*factorial(n-s))

    return K


def an_P(q, xq, E_1, E_2, E_0, R, kappa, a, N):

    PHI = zeros(len(q))
    for K in range(len(q)):
        rho = sqrt(sum(xq[K]**2))
        zenit = arccos(xq[K,2]/rho)
        azim  = arctan2(xq[K,1],xq[K,0])

        phi = 0.+0.*1j
        for n in range(N):
            for m in range(-n,n+1):
                P1 = lpmv(abs(m),n,cos(zenit))
#                if m<0:
#                    P1 *= (-1)**abs(m)*factorial(n-abs(m))/factorial(n+abs(m))

                Enm = 0.
                for k in range(len(q)):
                    rho_k   = sqrt(sum(xq[k]**2))
                    zenit_k = arccos(xq[k,2]/rho_k)
                    azim_k  = arctan2(xq[k,1],xq[k,0])
                    P2 = lpmv(abs(m),n,cos(zenit_k))
#                    if m<0:
#                        P2 *= (-1)**abs(m)*factorial(n-abs(m))/factorial(n+abs(m))

                    Enm += q[k]*rho_k**n*factorial(n-abs(m))/factorial(n+abs(m))*P2*exp(-1j*m*azim_k)
    
                C2 = (kappa*a)**2*get_K(kappa*a,n-1)/(get_K(kappa*a,n+1) + 
                        n*(E_2-E_1)/((n+1)*E_2+n*E_1)*(R/a)**(2*n+1)*(kappa*a)**2*get_K(kappa*a,n-1)/((2*n-1)*(2*n+1)))
                C1 = Enm/(E_2*E_0*a**(2*n+1)) * (2*n+1)/(2*n-1) * (E_2/((n+1)*E_2+n*E_1))**2

                if n==0 and m==0:
                    Bnm = Enm/(E_0*R)*(1/E_2-1/E_1) - Enm*kappa*a/(E_0*E_2*a*(1+kappa*a))
                else:
                    Bnm = 1./(E_1*E_0*R**(2*n+1)) * (E_1-E_2)*(n+1)/(E_1*n+E_2*(n+1)) * Enm - C1*C2

                phi += Bnm*rho**n*P1*exp(1j*m*azim)

        PHI[K] = real(phi)/(4*pi)

    return PHI
'''
q   = array([1.60217646e-19])
xq  = array([[1e-10,1e-10,0.]])
E_1 = 4.
E_2 = 80.
E_0 = 8.854187818e-12
R   = 1.
N   = 10
Q   = 1
Na  = 6.0221415e23
a   = R
kappa = 0.125

#PHI_sph = an_spherical(q, xq, E_1, E_2, E_0, R, N)
PHI_P = an_P(q, xq, E_1, E_2, E_0, R, kappa, a, N)

JtoCal = 4.184    
#E_solv_sph = 0.5*sum(q*PHI_sph)*Na*1e7/JtoCal
E_solv_P = 0.5*sum(q*PHI_P)*Na*1e7/JtoCal
#print 'With spherical harmonics: %f'%E_solv_sph
print 'With Legendre functions : %f'%E_solv_P
'''
