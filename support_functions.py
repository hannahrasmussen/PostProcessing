#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import time

from DES import Differential_Equation_Solver as DES
from nu_nu_coll import nu_nu_collisions as coll
from CollisionApprox import Collision_approx as ca
from Interpolate import interp


# In[2]:


alpha = 1/137
D = 1./1.79**3 
dtda_part2 = 2*np.pi/3
f_pi = 131 
Gf = 1.166*10**-11 
me = .511        
mpi_neutral = 135  
mpi_charged = 139.569  
mPL = 1.124*10**22 
mu = 105.661  
eps_e = me/mu
Enumax = (mu/2)*(1-(eps_e)**2)


# In[3]:


n = 10   
f_TINY = 1e-20
f_MINI = 1e-25
f_SMALL = 1e-30
f_BUFFER = 1e-40
MIN_eps_BUFFER = 12
a_MAXMULT = 2
x_values, w_values = np.polynomial.laguerre.laggauss(n)  
x_valuese, w_valuese = np.polynomial.legendre.leggauss(n)


# In[4]:


@nb.jit(nopython=True)
def I1(eps,x): #Energy Density
    numerator = (np.e**eps)*(eps**2)*((eps**2+x**2)**.5)
    denominator = np.e**((eps**2+x**2)**.5)+1
    return numerator/denominator

@nb.jit(nopython=True)
def I2(eps,x): #Pressure
    numerator = (np.e**eps)*(eps**4)
    denominator = ((eps**2+x**2)**.5)*(np.e**((eps**2+x**2)**.5)+1)
    return numerator/denominator

@nb.jit(nopython=True)
def dI1(eps,x): #Derivative of Energy Density
    numerator = (np.e**eps)*((eps**2+x**2)**.5)
    denominator = np.e**((eps**2+x**2)**.5)+1
    return (-x)*numerator/denominator

@nb.jit(nopython=True)
def dI2(eps,x): #Derivative of Pressure
    numerator = (np.e**eps)*3*(eps**2)
    denominator = ((eps**2+x**2)**.5)*(np.e**((eps**2+x**2)**.5)+1)
    return (-x)*numerator/denominator

@nb.jit(nopython=True)
def calculate_integral(I,x): #I is the function to integrate over, x is me/temp 
    return np.sum(w_values*I(x_values,x))  

@nb.jit(nopython=True)
def trapezoid(y_array,x_array):
    total = np.sum((x_array[1:]-x_array[:-1])*(y_array[1:]+y_array[:-1])/2)
    return total


# In[5]:


@nb.jit(nopython=True)
def rate1(ms,mixangle): 
    numerator = 9*(Gf**2)*alpha*(ms**5)*((np.sin(mixangle))**2)
    denominator = 512*np.pi**4
    Gamma = numerator/denominator
    return Gamma

@nb.jit(nopython=True)
def rate2(ms,mixangle):
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    part2 = ms*((ms**2)-(mpi_neutral**2))*(np.sin(mixangle))**2
    Gamma = part1*part2
    return Gamma

@nb.jit(nopython=True)
def rate3(ms,mixangle):
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    parentheses = ((ms**2) - (mpi_charged+me)**2)*((ms**2) - (mpi_charged-me)**2)
    part2 = ms * ((parentheses)**(1/2)) * (np.sin(mixangle))**2
    Gamma = part1*part2
    return 2*Gamma

@nb.jit(nopython=True)
def rate4(ms,mixangle):
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    parentheses = ((ms**2) - (mpi_charged+mu)**2)*((ms**2) - (mpi_charged-mu)**2)
    part2 = ms * ((parentheses)**(1/2)) * (np.sin(mixangle))**2
    Gamma = part1*part2
    return 2*Gamma 

@nb.jit(nopython=True)
def ts(ms,angle):
    return 1/(rate1(ms,angle)+rate2(ms,angle)+rate3(ms,angle)+rate4(ms,angle))

@nb.jit(nopython=True)
def ns(Tcm,t,ms,angle):
    part1 = D*3*1.20206/(2*np.pi**2)
    part2 = Tcm**3*np.e**(-t/ts(ms,angle))
    n_s = part1*part2
    return n_s


# In[6]:


@nb.jit(nopython=True)
def diracdelta(E,EI,i,E_arr):
    if i==0:
        bxR = E_arr[1] - E_arr[0]
        bxL = bxR
    elif len(E_arr)-i==1:
        bxL = E_arr[i] - E_arr[i-1]
        bxR = bxL
    else: 
        bxL = E_arr[i] - E_arr[i-1]
        bxR = E_arr[i+1] - E_arr[i]
    
    if EI - 0.6 * bxR <= E <= EI - 0.4 * bxR:
        x = EI - (E + 0.5 * bxR)
        A = 0.1 * bxR
        return 2/(bxL + bxR) * (0.5 + 0.75 / A**3 * (x**3 / 3 - A**2 * x))
    elif EI - 0.4 * bxR <= E <= EI + 0.4 * bxL:
        return 2 / (bxL + bxR)
    elif EI + 0.4 * bxL <= E <= EI + 0.6 * bxL:
        x = EI - (E - 0.5 * bxL)
        A = 0.1 * bxL
        return 2/(bxL + bxR) * (0.5 - 0.75 / A**3 * (x**3 / 3 - A**2 * x))
    else:
        return 0
    

@nb.jit(nopython=True)
def diracdelta2(E,EBmin,EBmax,E_B,gamma,v,i,E_arr): 
    if i==0:
        bxR = E_arr[1] - E_arr[0]
        bxL = bxR
    elif len(E_arr)-i==1:
        bxL = E_arr[i] - E_arr[i-1]
        bxR = bxL
    else: 
        bxL = E_arr[i] - E_arr[i-1]
        bxR = E_arr[i+1] - E_arr[i]
        
    r = 1/(2 * gamma * v * E_B)
    if EBmin - 0.5*bxR <= E <= EBmin:
        return r * (E - EBmin + 0.5 * bxR) * 2 / (bxR + bxL)
    elif EBmin <= E <= EBmin + 0.5*bxL:
        return r * (E - EBmin + 0.5 * bxR) * 2 / (bxR + bxL)
    elif EBmin + 0.5*bxL <= E <= EBmax - 0.5 * bxR:
        return r
    elif EBmax - 0.5* bxR <= E <= EBmax:
        return r * (EBmax - E + 0.5 * bxL) * 2 / (bxR + bxL)
    elif EBmax <= E <= EBmax + 0.5*bxL:
        return r * (EBmax - E + 0.5 * bxL) * 2 / (bxR + bxL)
    else:
        return 0

@nb.jit(nopython=True)
def EB(mA,mB,mC): 
    E_B = (mA**2 + mB**2 - mC**2)/(2*mA)
    return E_B

@nb.jit(nopython=True)
def Gammamua(a,b): #for both electron neutrinos and muon neutrinos for decay types III and IV
    if a>Enumax:
        return 0
    constant = 8*Gf*(mu**2)/(16*np.pi**3)
    part_b1 = (-1/4)*(me**4)*mu*np.log(abs(2*b-mu))
    part_b2 = (-1/6)*b
    part_b3 = 3*(me**4)+6*(me**2)*mu*b
    part_b4 = (mu**2)*b*(4*b-3*mu)
    part_b = (part_b1+part_b2*(part_b3+part_b4))/mu**3
    part_a1 = (-1/4)*(me**4)*mu*np.log(abs(2*a-mu))
    part_a2 = (-1/6)*a
    part_a3 = 3*(me**4)+6*(me**2)*mu*a
    part_a4 = (mu**2)*a*(4*a-3*mu)
    part_a = (part_a1+part_a2*(part_a3+part_a4))/mu**3
    integral = part_b-part_a
    Gam_mua = constant*integral
    return Gam_mua

@nb.jit(nopython=True)
def Gammamub(): #for both electron neutrinos and muon neutrinos for decay types III and IV 
    constant = 8*Gf*(mu**2)/(16*np.pi**3)
    part_a1 = 3*(me**4)*(mu**2)*np.log(abs(2*Enumax-mu))
    part_a2 = 6*(me**4)*Enumax*(mu+Enumax)
    part_a3 = 16*(me**2)*mu*(Enumax**3)
    part_a4 = 4*(mu**2)*(Enumax**3)*(3*Enumax - 2*mu)
    part_a5 = 24*mu**3
    part_b1 = 3*(me**4)*(mu**2)*np.log(abs(-mu))/part_a5
    integral = ((part_a1+part_a2+part_a3+part_a4)/part_a5)-part_b1
    Gam_mub = -1*constant*integral
    return Gam_mub

Gam_mub = Gammamub()

@nb.jit(nopython=True)
def u_integral(E_mumin,E_mumax,Eactive):
    Eu_array = ((E_mumax-E_mumin)/2)*x_valuese + ((E_mumax+E_mumin)/2)
    integral = 0
    for i in range(n):
        gammau = Eu_array[i]/mu
        pu = (Eu_array[i]**2-mu**2)**(1/2)
        vu = pu/Eu_array[i]
        Gam_mua = Gammamua(Eactive/(gammau*(1+vu)),min(Enumax,Eactive/(gammau*(1-vu))))
        integral = integral + (w_valuese[i]*((E_mumax-E_mumin)/2)*(1/(2*gammau*vu))*Gam_mua)
    return integral


# In[7]:


@nb.jit(nopython=True)
def C_round(j,f,p):
    c,c_frs = coll.cI(j,f,p)
    if abs(c/c_frs) < 3e-15:
        return 0
    else:
        return c

@nb.jit(nopython=True,parallel=True)
def C_short(p,f,T,k):
    c = np.zeros(len(p))
    
    for i in nb.prange(1,len(p)-1):
        if i < k[0]:
            c[i] = C_round(i,f[:k[1]],p[:k[1]])
        elif i < k[1]:
            c[i] = C_round(i,f[:k[2]],p[:k[2]])
        else:
            c[i] = C_round(i,f,p)
    return c

def find_breaks(f, E2_index=0, E1_index=0):
    if (len(np.where(f < f_TINY)[0]) > 0):
        k_0 = np.where(f < f_TINY)[0][0]
    else: 
        k_0 = len(f) - 1
    if (len(np.where(f < f_MINI)[0]) > 0):
        k_1 = np.where(f < f_MINI)[0][0]
    else:
        k_1 = len(f) - 1
    if (len(np.where(f < f_SMALL)[0]) > 0):
        k_2 = np.where(f < f_SMALL)[0][0]
    else:
        k_2 = len(f) - 1
    
    for i in range(k_0, len(f)):
        if f[i] > f_TINY:
            k_0 = i+1
    for i in range(k_1,len(f)):
        if f[i] > f_MINI:
            k_1 = i+1
    for i in range(k_2,len(f)):
        if f[i] > f_SMALL:
            k_2 = i+1
            
    Echeck = [E2_index, E1_index]
    k_return = [k_0, k_1, k_2]
    for j in range(3):
        for i in range(2):
            if Echeck[i] - MIN_eps_BUFFER < k_return[j] <= Echeck[i]:
                k_return[j] += 2 * MIN_eps_BUFFER
            if Echeck[i] <= k_return[j] < Echeck[i] + MIN_eps_BUFFER:
                k_return[j] += MIN_eps_BUFFER
        for jj in range(j+1,3):
            if k_return[jj] < k_return[j] + MIN_eps_BUFFER:
                k_return[jj] = k_return[j] + MIN_eps_BUFFER
        if k_return[j] >= len(f):
            k_return[j] = len(f) - 1

    return k_return


# In[8]:


#def make_collision_fn(ms, mixangle, e_array, A_model, n_model, kk):
def make_collision_fn(ms, mixangle, e_array, A_model, n_model):
    
    d1r = rate1(ms,mixangle) 
    d2r = rate2(ms,mixangle) 
    d3r = rate3(ms,mixangle) 
    d4r = rate4(ms,mixangle)

    E_B1 = ms/2
    E_B2 = (ms**2 - mpi_neutral**2)/(2*ms)
    
    #constants referring to initial part of decay 3:
    E_pi3 = EB(ms,mpi_charged,me) 
    p_pi3 = (E_pi3**2 - mpi_charged**2)**(1/2) 
    v3 = p_pi3/E_pi3
    gammapi3 = E_pi3/mpi_charged
    
    #constants referring to decay 3a:
    E_B3 = EB(mpi_charged,0,mu)
    E_B3max = gammapi3*E_B3*(1+v3)
    E_B3min = gammapi3*E_B3*(1-v3)
                        
    #additional constants referring to decay 3b:
    E_mu3 = EB(mpi_charged,mu,0) 
    p_mu3 = (E_mu3**2 - mu**2)**(1/2) 
    E_mumax3 = gammapi3*(E_mu3 + (v3*p_mu3))
    E_mumin3 = gammapi3*(E_mu3 - (v3*p_mu3))
    
    #constants referring the initial decay of decay 4:
    E_pi4 = EB(ms,mpi_charged,mu) 
    p_pi4 = (E_pi4**2 - mpi_charged**2)**(1/2)
    v4 = p_pi4/E_pi4
    gammapi4 = E_pi4/mpi_charged
    Eu = ms-E_pi4 
    
    #constants referring to decay 4b:
    E_B4 = EB(mpi_charged,0,mu)
    E_B4max = gammapi4*E_B4*(1 + v4)
    E_B4min = gammapi4*E_B4*(1 - v4)
    
    #constants referring to decay 4c:
    E_mu4 = EB(mpi_charged,mu,0)
    p_mu4 = (E_mu4**2 - mu**2)**(1/2) 
    E_mumax4 = gammapi4*(E_mu4 + (v4*p_mu4))
    E_mumin4 = gammapi4*(E_mu4 - (v4*p_mu4))
    
    #@nb.jit(nopython=True)
    def C_ve(p_array, Tcm, T, f):
        C_array = p_array**n_model * (f - ca.f_eq(p_array, T, 0))
        return - A_model * ca.n_e(T) * Gf**2 * T**(2-n_model) * C_array
    
    #@nb.jit(nopython=True)
    def derivatives(a,y): 
        num = len(y)
        d_array = np.zeros(len(y))
        Tcm = 1/a 

        dtda_part1 = mPL/(2*a)
        dtda_part3 = (y[-2]**4*np.pi**2)/15
        dtda_part4 = 2*y[-2]**4*calculate_integral(I1,me/y[-2])/np.pi**2
        dtda_part6 = ms*ns(Tcm,y[-1],ms,mixangle)
        dtda_part7 = (Tcm**4/(2*np.pi**2))*trapezoid(y[:int(num-3)]*e_array[:int(num-3)]**3,e_array[:int(num-3)])
        dtda = dtda_part1/(dtda_part2*(dtda_part3+dtda_part4+dtda_part6+dtda_part7))**.5
        d_array[-1] = dtda

        #df/da for the neutrinos and antineutrinos at epsilon = 0:
        d3b_e0 = 2*(1-eps_e**2)*d3r*gammapi3*(mu**2)*(Gf**2)*E_mu3*ns(Tcm,y[-1],ms,mixangle)*dtda/(np.pi*Gam_mub)
        d4b_e0 = 2*(1-eps_e**2)*d4r*(Eu/mu)*(mu**2)*(Gf**2)*ns(Tcm,y[-1],ms,mixangle)*dtda/(np.pi*Gam_mub)
        d4c_e0 = 2*(1-eps_e**2)*d4r*gammapi4*(mu**2)*(Gf**2)*E_mu4*ns(Tcm,y[-1],ms,mixangle)*dtda/(np.pi*Gam_mub)
        d_array[0] = d3b_e0+d4b_e0+d4c_e0

        #if kk[0] == 0:
        #    c = coll.C(e_array*Tcm,y[:num-3]) 
        #else:
        #    c = C_short(e_array*Tcm,y[:num-3],y[-2],kk) 
        c = coll.C(e_array*Tcm,y[:num-3]) 
        c += C_ve(e_array*Tcm, Tcm, y[-2], y[:num-3])
        c *= dtda
            
        for i in range (1,num-3): 
            eps = e_array[i]
            coefficient = (2*np.pi**2)/(eps**2*Tcm**2*a**3)
            d1 = d1r*diracdelta((eps*Tcm),E_B1,i,e_array*Tcm)
            d2 = d2r*diracdelta((eps*Tcm),E_B2,i,e_array*Tcm)
            d3a = .5*d3r*diracdelta2(eps*Tcm,E_B3min,E_B3max,E_B3,gammapi3,v3,i,e_array*Tcm)
            d3b = (d3r/(2*gammapi3*v3*p_mu3*Gam_mub))*u_integral(E_mumin4,E_mumax4,eps*Tcm)
            Gam_mua = Gammamua((eps*Tcm)/(gammapi4*(1+v4)),min(Enumax,(eps*Tcm)/(gammapi4*(1-v4))))
            d4a = (d4r/(2*gammapi4*v4))*(Gam_mua/Gam_mub)
            d4b = .5*d4r*diracdelta2((eps*Tcm),E_B4min,E_B4max,E_B4,gammapi4,v4,i,e_array*Tcm)
            d4c = (d4r/(2*gammapi4*v4*p_mu4*Gam_mub))*u_integral(E_mumin4,E_mumax4,eps*Tcm)
                
            d_array[i] = coefficient*(d1+d2+d3a+d3b+d4a+d4b+d4c)*ns(Tcm,y[-1],ms,mixangle)*a**3*dtda + c[i] #neutrinos only, antineutrinos not included

        df_array = d_array[:-3]*e_array**3/(2*np.pi**2) 
        dQda_part1 = ms*ns(Tcm,y[-1],ms,mixangle)*a**3*dtda/ts(ms,mixangle)
        dQda_part2 = Tcm**4*a**3*trapezoid(df_array,e_array)
        dQda = dQda_part1-dQda_part2
        d_array[-3] = dQda

        dTda_constant1 = (4*np.pi**2/45)+(2/np.pi**2)*(calculate_integral(I1,me/y[-2]) + (1/3)*calculate_integral(I2,me/y[-2]))
        dTda_constant2 = 2*me*y[-2]*a**3/(np.pi**2)
        dTda_numerator1 = -3*a**2*y[-2]**3*dTda_constant1
        dTda_numerator2 = dQda/y[-2]
        dTda_denominator = (3*y[-2]**2*a**3*dTda_constant1) - (dTda_constant2*(calculate_integral(I1,me/y[-2]) - (1/3)*calculate_integral(I2,me/y[-2])))
        dTda = (dTda_numerator1 + dTda_numerator2)/dTda_denominator
        d_array[-2] = dTda

        return d_array, c
    
    return derivatives

@nb.jit(nopython=True)
def time_derivative(a,T): 
        Tcm = 1/a 
        dtda_part1 = mPL/(2*a)
        dtda_part2 = 2*np.pi/3
        dtda_part3 = (T**4*np.pi**2)/15
        dtda_part4 = 2*T**4*calculate_integral(I1,me/T)/np.pi**2
        dtda_part5 = (7*np.pi**2/40)*(1/a)**4
        dtda = dtda_part1/(dtda_part2*(dtda_part3+dtda_part4+dtda_part5))**.5
        return dtda
