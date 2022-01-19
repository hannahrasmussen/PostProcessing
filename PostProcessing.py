#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import os
from matplotlib import animation
from scipy.interpolate import CubicSpline as cs
from CollisionApprox import Collision_approx as ca

import support_functions as sf
from N2P import N2P_calc as N2P


#In[2]:


hbar = 6.58212e-22
mpi_neutral = 135  

def f_nu(E,Tcm):
    return 1/(np.exp(E/Tcm)+1)

eq_a_arr = np.load('../../eq_arrs/eq_arrs/Luke_a.npy')
eq_T_arr = np.load('../../eq_arrs/eq_arrs/Luke_T.npy')
eq_eta_arr = np.load('../../eq_arrs/eq_arrs/Luke_eta.npy')
eq_e_arr = np.linspace(0,200,201)
cs_eta = cs(np.flip(eq_T_arr),np.flip(eq_eta_arr))
cs_Tcm = cs(np.flip(eq_T_arr),np.flip(1/eq_a_arr))
n2p_st = np.zeros(len(eq_a_arr))
p2n_st = np.zeros(len(eq_a_arr))
Hub_st = np.zeros(len(eq_a_arr))
for i in range(len(eq_a_arr)):
    eq_f_arr = f_nu(eq_e_arr/eq_a_arr[i],1/eq_a_arr[i])
    n2p_st[i] = N2P.varlam_np(eq_a_arr[i], eq_e_arr, eq_eta_arr[i], eq_f_arr, eq_T_arr[i])
    p2n_st[i] = N2P.varlam_pn(eq_a_arr[i], eq_e_arr, eq_eta_arr[i], eq_f_arr, eq_T_arr[i])
for i in range(1,len(eq_a_arr)):
    dtda = sf.time_derivative(eq_a_arr[i],eq_T_arr[i])
    Hub_st[i] = (dtda * eq_a_arr[i])**(-1)
Hub_st[0] = Hub_st[1]


#In[3]:


def create_movie_arrays(ms,mixangle):
    folder = "../../{}-{:.4}-FullTestNew/{}-{:.4}-FullTestNew".format(ms,mixangle,ms,mixangle)
    data = np.load(folder + "/mass_{}_mix_{:.4}.npz".format(ms,mixangle), allow_pickle=True)
    a_tot = data['scalefactors']
    e_tot = data['e']
    f_tot = data['fe']
    T_tot = data['temp']
    t_tot = data['time']
    Num_steps = len(a_tot)
    #d_tot = data['decayrate']
    #c_tot = data['collisionrate']
    #n2p = data['n_p_rate']
    #p2n = data['p_n_rate']
    
    eq_eta = cs_eta(T_tot)
    max_e_len = 1
    for i in range(len(e_tot)):
        max_e_len = max(len(e_tot[i]),max_e_len)
    d_tot = np.zeros((Num_steps,max_e_len+3))
    c_tot = np.zeros((Num_steps,max_e_len))
    n2p = np.zeros(Num_steps)
    p2n = np.zeros(Num_steps)
    Hub = np.zeros(Num_steps)
    
    for i in range(Num_steps):
        len_e = len(np.where(e_tot[i] > 0)[0])+1
        n2p[i] = N2P.varlam_np(a_tot[i], e_tot[i][:len_e], eq_eta[i], f_tot[i][:len_e], T_tot[i])
        p2n[i] = N2P.varlam_pn(a_tot[i], e_tot[i][:len_e], eq_eta[i], f_tot[i][:len_e], T_tot[i])
    
    for i in range(Num_steps):
        #if i == 0:
        #    len_e = len(np.where(e_tot[0] > 0)[0])+1
        #    A_model, n_model = ca.model_An(a_tot[0], T_tot[0], 1/(0.9 * a_tot[0] * T_tot[0]), e_tot[0][:len_e], f_tot[0][:len_e])
        #    kk = [0,0,0]
        #    fn = sf.make_collision_fn(ms,mixangle,e_tot[0][:len_e],A_model,n_model,kk)
        #else:
        #    len_e = len(np.where(e_tot[i] > 0)[0])+1
        #    A_model, n_model = ca.model_An(a_tot[i], T_tot[i], 1/(0.9 * a_tot[i] * T_tot[i]), e_tot[i][:len_e], f_tot[i][:len_e])
        #    E_B1 = ms/2
        #    E_B2 = (ms**2 - mpi_neutral**2)/(2*ms)
        #    E2_index = np.where(e_tot[i] < E_B2 * a_tot[i])[0][-1]
        #    E1_index = np.where(e_tot[i] < E_B1 * a_tot[i])[0][-1]
        #    kk = sf.find_breaks(f_tot[i][:len_e], E2_index, E1_index)
        #    fn = sf.make_collision_fn(ms,mixangle,e_tot[i][:len_e],A_model,n_model,kk)
        
        len_e = len(np.where(e_tot[i] > 0)[0])+1
        A_model, n_model = ca.model_An(a_tot[i], T_tot[i], 1/(0.9 * a_tot[i] * T_tot[i]))
        fn = sf.make_collision_fn(ms,mixangle,e_tot[i][:len_e],A_model,n_model)
        y_temp = np.zeros(len_e+3)
        y_temp[:len_e] = f_tot[i][:len_e]
        y_temp[-2] = T_tot[i]
        y_temp[-1] = t_tot[i]
        d, c = fn(a_tot[i],y_temp)
        d_tot[i,:len_e+3] = d
        c_tot[i,:len_e] = c
        Hub[i] = (d[-1] * a_tot[i])**(-1)
        
    Hub[0] = Hub[1] 
    
    np.savez(folder+'/movie_arrays', a=a_tot[:Num_steps], e=e_tot[:Num_steps], f=f_tot[:Num_steps], T=T_tot[:Num_steps], t=t_tot[:Num_steps], d=d_tot, c=c_tot, np=n2p, np_SM=n2p_st, pn=p2n, pn_SM=p2n_st, Hub=Hub, Hub_SM=Hub_st)
    return a_tot, e_tot, f_tot, T_tot, t_tot, d_tot, c_tot, n2p, n2p_st, p2n, p2n_st, Hub, Hub_st


#In[4]:


def check_int(a_arr, e_mat, f_mat):
    output = np.zeros(len(a_arr))
    
    for i in range(len(a_arr)):
        e_arr, f_arr = N2P.set_arrs(a_arr[i],e_mat[i],f_mat[i])
        integrand = e_arr/a_arr[i] 
        output[i] = N2P.newtrapezoid(integrand, e_arr)
        
    return output

def pp(ms, mixangle):
    folder = "../../{}-{:.4}-FullTestNew/{}-{:.4}-FullTestNew".format(ms,mixangle,ms,mixangle)
    data = np.load(folder + "/mass_{}_mix_{:.4}.npz".format(ms,mixangle), allow_pickle=True)
    a_arr = data['scalefactors']
    e_mat = data['e']
    f_mat = data['fe']
    T_arr = data['temp']
    t_arr = data['time']
    
    N_eff = N2P.N_eff(T_arr[0],T_arr[-1],a_arr[0],a_arr[-1],f_mat[-1],e_mat[-1],e_mat[-1][1]-e_mat[-1][0]) #This won't work w/ varying boxsize...
    dilution_factor = N2P.F(T_arr[0],T_arr[-1],a_arr[0],a_arr[-1]) 
    spb_array = N2P.spb(a_arr,T_arr) 
    n2p, p2n, Hubble = N2P.driver(a_arr, e_mat, f_mat, T_arr, t_arr, ms, mixangle) 
    std_cosmo_spb = np.zeros(len(T_arr)) + 5.9*10**9 #entropy per baryon in standard cosmology
    t_YnYp,Y_arr = N2P.YnYp(n2p,p2n,T_arr,t_arr)
    
    print("N effective is " + str(int(100*N_eff)/100))
    print("The dilution factor is " + str(int(100*dilution_factor)/100))
    
    plt.figure(figsize=(8,8))
    plt.loglog(T_arr,n2p/Hubble,label='$n\Rightarrow p$')
    plt.loglog(T_arr,p2n/Hubble,label='$p\Rightarrow n$')
    plt.loglog(T_arr,Hubble, linestyle='--',label = 'Hubble rate')
    plt.xlabel('$T$ (MeV)',fontsize=16)
    plt.ylabel('Rate ($s^{-1}$)',fontsize=16)
    plt.xlim(10,10**-2)
    plt.ylim((10**-10,10**5))
    plt.legend(loc="upper right", fontsize=14)
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.show()
    
    plt.figure(figsize=(8,8))
    plt.loglog(1/a_arr,T_arr, linestyle="-", color='purple', label = "Our model")
    plt.loglog(1/eq_a_arr,eq_T_arr, linestyle="--", color='gold', label = "Std. Comsmology")
    plt.loglog(1/a_arr,1/a_arr, linestyle=':', color='grey')
    plt.xlabel("Tcm (MeV)",fontsize=16)
    plt.ylabel("T (MeV)",fontsize=16)
    plt.xlim(15,0.001)
    plt.ylim(0.001,15)
    plt.legend(loc="upper right", fontsize=14)
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.show()
    
    plt.figure(figsize=(8,8))
    plt.semilogx(T_arr,spb_array*10**-9,color='purple',label="Our model")
    plt.semilogx(T_arr,std_cosmo_spb*10**-9,color='gold',linestyle="--", label="Std. Cosmology")
    plt.xlabel("Plasma Temperature (MeV)",fontsize=16)
    plt.ylabel("Entropy-per-baryon ($\\times 10^9 {\\rm } k_{\\rm B}$)",fontsize=16)
    plt.xlim(15,0.01)
    plt.legend(loc="lower right", fontsize=14)
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.show()

    plt.figure(figsize=(8,6))
    plt.semilogx(t_YnYp,Y_arr[0],label = '$Y_n$',color='darkturquoise')
    plt.semilogx(t_YnYp,Y_arr[1],label = '$Y_p$',color='hotpink')
    plt.semilogx(t_YnYp,Y_arr[2],label = '$Y_n$ + $Y_p$',color='blue')
    plt.xlabel('Time (s)',fontsize=18)
    plt.ylabel('Abundance',fontsize=18)
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.show()

def inspect_graphs(ms,mixangle):
    folder = "../../{}-{:.4}-FullTestNew/{}-{:.4}-FullTestNew".format(ms,mixangle,ms,mixangle)
    data = np.load(folder + "/mass_{}_mix_{:.4}.npz".format(ms,mixangle), allow_pickle=True)
    a_arr = data['scalefactors']
    e_mat = data['e']
    f_mat = data['fe']
    T_arr = data['temp']
    t_arr = data['time']
    
    n2p, p2n, Hubble = N2P.driver(a_arr, e_mat, f_mat, T_arr, t_arr, ms, mixangle)
    j = 10**-10
    for i in range (len(t_arr)-1): 
        if t_arr[i+1]<=t_arr[i]:
            t_arr[i+1] = t_arr[i+1] + j*(1.52*10**21)
            j = j + 10**-10
    
    t_YnYp, results_YnYp = N2P.YnYp(n2p,p2n,T_arr,t_arr)
    
    cs_n2p = cs(t_arr/(1.52*10**21),n2p) 
    cs_p2n = cs(t_arr/(1.52*10**21),p2n)
    t_cs = np.logspace(16,23,10000)/(1.52*10**21)
    n2p_cs = cs_n2p(t_cs)
    p2n_cs = cs_p2n(t_cs)
    
    nue_n, pos_n, n, p_elec, anue_p, anue_elec_p = N2P.individual_driver(a_arr, e_mat, f_mat, T_arr, ms, mixangle)
    integral = check_int(a_arr, e_mat, f_mat)
    
    plt.figure(figsize=(8,6))
    plt.semilogx(t_YnYp,results_YnYp[0],label = '$Y_n$',color='darkturquoise')
    plt.semilogx(t_YnYp,results_YnYp[1],label = '$Y_p$',color='hotpink')
    plt.semilogx(t_YnYp,results_YnYp[2],label = '$Y_n$ + $Y_p$',color='blue')
    plt.xlabel('Time (s)',fontsize=18)
    plt.ylabel('Abundance',fontsize=18)
    plt.title('Relative abundances',fontsize=18)
    plt.xlim(10**-3,10**2)
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.show()
    
    plt.figure(figsize=(8,6))
    plt.loglog(t_YnYp[105:160],results_YnYp[0][105:160],label = '$Y_n$',color='darkturquoise')
    plt.loglog(t_YnYp[105:160],results_YnYp[1][105:160],label = '$Y_p$',color='hotpink')
    plt.loglog(t_YnYp[105:160],results_YnYp[2][105:160],label = '$Y_n$ + $Y_p$',color='blue')
    plt.xlabel('Time (s)',fontsize=18)
    plt.ylabel('Abundance',fontsize=18)
    plt.title('Zoom-in',fontsize=18)
    #plt.xlim(10**-1,10**1)
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)
    plt.legend(loc="upper left", fontsize=14)
    plt.show()
    
    plt.figure(figsize=(8,8))
    plt.loglog(t_arr/(1.52*10**21),n2p,label='$n\Rightarrow p$')
    plt.loglog(t_arr/(1.52*10**21),p2n,label='$p\Rightarrow n$')
    plt.loglog(t_arr/(1.52*10**21),Hubble, linestyle='--',label = 'Hubble rate')
    plt.xlabel('Time (s)',fontsize=16)
    plt.ylabel('Rate ($s^{-1}$)',fontsize=16)
    plt.title('Rates from data',fontsize=18)
    plt.xlim(10**-3,10**2)
    plt.ylim((10**-10,10**6))
    plt.legend(loc="upper right", fontsize=14)
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.show()
    
    plt.figure(figsize=(8,8))
    plt.loglog(t_cs,n2p_cs,label='$n\Rightarrow p$')
    plt.loglog(t_cs,p2n_cs,label='$p\Rightarrow n$')
    plt.xlabel('Time (s)',fontsize=16)
    plt.ylabel('Rate ($s^{-1}$)',fontsize=16)
    plt.title('Cubic-splined rates',fontsize=18)
    plt.xlim(10**-3,10**2)
    plt.ylim((10**-10,10**6))
    plt.legend(loc="upper right", fontsize=14)
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.show()
    
    plt.figure(figsize=(8,8))
    plt.loglog(T_arr,nue_n,label='$\\nu_e + n \\to p + e^- $',color='red')
    plt.loglog(T_arr,pos_n,label='$e^+ + n \\to p + \\bar{\\nu}_e$',color='orange')
    plt.loglog(T_arr,n,label = '$n \\to \\bar{\\nu}_e + e^- + p $',color='yellow')
    plt.loglog(T_arr,p_elec,label = '$p + e^- \\to \\nu_e + n $',color='green')
    plt.loglog(T_arr,anue_p,label = '$p + \\bar{\\nu}_e \\to e^+ + n$',color='blue')
    plt.loglog(T_arr,anue_elec_p,label = '$\\bar{\\nu}_e + e^- + p \\to n $',color='purple')
    plt.xlabel('$T$ (MeV)',fontsize=16)
    plt.ylabel('Rate ($s^{-1}$)',fontsize=16)
    plt.title('Break down by rxn',fontsize=18)
    plt.xlim(10,10**-2)
    plt.ylim((10**-10,10**6))
    plt.legend(loc="upper right", fontsize=14)
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.show()
    
    plt.figure(figsize=(8,8))
    for i in range(len(e_mat)-1):
        if (True != np.array_equiv(e_mat[i+1],e_mat[i])):
            plt.scatter(T_arr[i],integral[i],color='blue')
    plt.loglog(T_arr,integral,color='red')
    plt.xlabel('$T$ (MeV)',fontsize=16)
    plt.ylabel('Integral over p',fontsize=16)
    plt.title('Blue dots are where e_arr changes',fontsize=14)
    plt.xlim(10**1,10**-2)
    #plt.ylim((10**2,10**6))
    #plt.legend(loc="lower left", fontsize=14)
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.show()


# In[5]:


def graph_f(ms,mixangle,save=False):
    fn = "../../{}-{:.4}-FullTestNew/{}-{:.4}-FullTestNew".format(ms,mixangle,ms,mixangle)
    fn_f = fn + '/' + "f.pdf"
    if not os.path.isfile(fn+'/movie_arrays.npz'):
        print("Need to create movie arrays.  Will take a few minutes.")
        create_movie_arrays(ms,mixangle)
    arrs = np.load(fn+'/movie_arrays.npz', allow_pickle=True)
    ts = sf.ts(ms,mixangle)*hbar
    eq_Tcm = cs_Tcm(arrs['T']) 
    
    e_vec = arrs['e'][-1]
    f_vec = arrs['f'][-1]
    a = arrs['a'][-1]
    len_e = len(np.where(e_vec > 0)[0])+1
    eps = e_vec[:len_e]
    fv = f_vec[:len_e]

    e_thermal = eps**3 / (np.exp(eps / a /  eq_Tcm[-1]) +1) / (2 * np.pi**3)
    nu = eps**3 * fv / (2 * np.pi**3)
    
    plt.figure(figsize=(10,6))
    plt.axes([0.2,0.12,0.7,0.85])
    plt.loglog(eps, e_thermal, linewidth=2, linestyle='--',color='b')
    plt.loglog(eps, nu, linewidth=2, color='k')
    plt.xlim([0.5,3000])
    plt.ylim([1e-10,10])
    plt.xticks([1e0,1e1,1e2,1e3])
    plt.yticks([1e-9,1e-6,1e-3,1e0])
    plt.xlabel(r"$\epsilon = E_\nu / T_{\rm cm}$",fontsize=20)
    plt.ylabel(r"$\epsilon^3 \, f(\epsilon)$",fontsize=20)
    #plt.title("{} MeV, {:.3} s decay lifetime".format(ms,ts))
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)
    plt.plot()
    if save:
        plt.savefig(fn_f)

def graph_df(ms,mixangle,save=False):
    fn = "../../{}-{:.4}-FullTestNew/{}-{:.4}-FullTestNew".format(ms,mixangle,ms,mixangle)
    fn_df = fn + '/' + "df.pdf"
    
    if not os.path.isfile(fn+'/movie_arrays.npz'):
        print("Need to create data arrays.  Will take a few minutes.")
        create_movie_arrays(ms,mixangle)
    arrs = np.load(fn+'/movie_arrays.npz', allow_pickle=True)
    ts = sf.ts(ms,mixangle)*hbar
    eq_Tcm = cs_Tcm(arrs['T']) 
    
    e_vec = arrs['e'][-1][:]
    d_vec = arrs['d'][-1][:]
    c_vec = arrs['c'][-1][:]
    T = arrs['T'][-1]
    t = arrs['t'][-1]
    a = arrs['a'][-1]
    len_e = len(np.where(e_vec > 0)[0])+1

    eps = e_vec[:len_e]
    d = d_vec[:len_e]
    c = c_vec[:len_e]
    
    plt.figure(figsize=(10,6))
    plt.axes([0.2,0.12,0.7,0.85])
    plt.loglog(eps, d*a, linewidth=2, color='k')
    plt.loglog(eps, c*a, linewidth=2, color='k', linestyle='--')
    plt.loglog(eps, -c*a, linewidth=2, color='r', linestyle='--')
    plt.xlim([0.5,3000])
    plt.ylim([1e-10,1e4])
    plt.yticks([1e-10,1e-5,1e0])
    plt.xlabel(r"$\epsilon = E_\nu / T_{\rm cm}$",fontsize=20)
    plt.ylabel(r"Rate, $(df/dt) / H$",fontsize=20)
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)
    plt.show()
    if save:
        plt.save(fn_df)
    
def graph_TTcm(ms,mixangle,save=False):
    fn = "../../{}-{:.4}-FullTestNew/{}-{:.4}-FullTestNew".format(ms,mixangle,ms,mixangle)
    fn_TTcm = fn + '/' + "TTcm.pdf"
    
    if not os.path.isfile(fn+'/movie_arrays.npz'):
        print("Need to create movie arrays.  Will take a few minutes.")
        create_movie_arrays(ms,mixangle)
    arrs = np.load(fn+'/movie_arrays.npz', allow_pickle=True)
    eq_Tcm = cs_Tcm(arrs['T']) 
    
    a_vec = arrs['a'][:(-1)]
    T_vec = arrs['T'][:(-1)]
        
    plt.figure(figsize=(10,6))
    plt.axes([0.2,0.12,0.7,0.85])
    plt.semilogx(T_vec,T_vec*a_vec,color='b')
    plt.semilogx(T_vec,T_vec / eq_Tcm[:(-1)],linestyle='--',color='k')
    plt.xlim(15,0.015)
    plt.ylim(0.95,3)
    plt.xlabel(r"$T$ (MeV)",fontsize=20)
    plt.ylabel(r"$T / T_{\rm cm}$",fontsize=20)
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)
    plt.show()
    if save:
        plt.save(fn_TTcm)
    
def graph_n2p(ms,mixangle,save=False):
    fn = "../../{}-{:.4}-FullTestNew/{}-{:.4}-FullTestNew".format(ms,mixangle,ms,mixangle)
    fn_n2p = fn + '/' + "n2p.pdf"
    
    if not os.path.isfile(fn+'/movie_arrays.npz'):
        print("Need to create movie arrays.  Will take a few minutes.")
        create_movie_arrays(ms,mixangle)
    arrs = np.load(fn+'/movie_arrays.npz', allow_pickle=True) 
    
    T_vec = arrs['T']
    Hub = arrs['Hub']
    Hub_SM = arrs['Hub_SM']
    n2p = arrs['np'] / Hub
    n2p_SM = arrs['np_SM'] / Hub_SM
    p2n = arrs['pn'] / Hub
    p2n_SM = arrs['pn_SM'] / Hub_SM
    
    plt.figure(figsize=(10,6))
    plt.axes([0.2,0.12,0.70,0.85])
    plt.loglog(T_vec,n2p,color='salmon',linewidth=2,label=r"$\lambda_{n \rightarrow p}$")
    plt.loglog(eq_T_arr,n2p_SM,color='salmon',linewidth=1,linestyle='--')  
    plt.loglog(T_vec,p2n,color='rebeccapurple',linewidth=2,label=r"$\lambda_{p \rightarrow n}$")
    plt.loglog(eq_T_arr,p2n_SM,color='rebeccapurple',linewidth=1,linestyle='--')
    plt.axhline(1,color='0.50',linestyle='-.')
    plt.xlim(15,0.015)
    plt.ylim(1e-5,1e5)
    plt.yticks([1e-3,1e0,1e3])
    plt.xlabel(r"$T$ (MeV)",fontsize=20)
    plt.ylabel(r"$\lambda / H$",fontsize=20)
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)
    #plt.text(10,4000,label=r"$\lambda_{n \rightarrow p}$",color='purple',fontsize=14)
    #plt.text(10,10000,label=r"$\lambda_{p \rightarrow n}$",color='gold',fontsize=14)
    plt.legend(loc="lower left",fontsize=14)
    plt.show()
    if save:
        plt.save(fn_n2p)
    
def make_graphs(ms,mixangle,save=False):
    graph_f(ms,mixangle,save)
    graph_df(ms,mixangle,save)
    graph_TTcm(ms,mixangle,save)
    graph_n2p(ms,mixangle,save)


#In[6]:


def movie_f(ms,mixangle):
    fn = "../../{}-{:.4}-FullTestNew/{}-{:.4}-FullTestNew".format(ms,mixangle,ms,mixangle)
    fn_f = fn + '/' + "f.mp4"
    if not os.path.isfile(fn+'/movie_arrays.npz'):
        print("Need to create movie arrays.  Will take a few minutes.")
        create_movie_arrays(ms,mixangle)
    arrs = np.load(fn+'/movie_arrays.npz', allow_pickle=True)
    ts = sf.ts(ms,mixangle)*hbar
    eq_Tcm = cs_Tcm(arrs['T']) 
    
    def init():
        e_thermal.set_data([], [])
        nu.set_data([], [])
        return e_thermal, nu
    
    def animate(i):
        e_vec = arrs['e'][i]
        f_vec = arrs['f'][i]
        T = arrs['T'][i]
        t = arrs['t'][i]
        a = arrs['a'][i]
        len_e = len(np.where(e_vec > 0)[0])+1
        eps = e_vec[:len_e]
        fv = f_vec[:len_e]

        e_thermal.set_data(eps, eps**3 / (np.exp(eps / a /  eq_Tcm[i]) +1) / (2 * np.pi**3))
        nu.set_data(eps, eps**3 * fv / (2 * np.pi**3))
        
        return e_thermal, nu
    
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes([0.2,0.12,0.7,0.85])
    e_thermal, = ax.loglog([], [], linewidth=2, linestyle='--',color='b')
    nu, = ax.loglog([], [], linewidth=2, color='k')
    ax.set_ylim([1e-10,10])
    ax.set_yticks([1e-9,1e-6,1e-3,1e0])
    ax.set_xlim([0.5,3000])
    ax.set_xticks([1e0,1e1,1e2,1e3])
    ax.set_xlabel(r"$\epsilon = E_\nu / T_{\rm cm}$",fontsize=20)
    ax.set_ylabel(r"$\epsilon^3 \, f(\epsilon)$",fontsize=20)
    #ax.set_title("{} MeV, {:.3} s decay lifetime".format(ms,ts))
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = len(arrs['a']), interval = 33, blit=True)
    anim.save(fn_f)

def movie_df(ms,mixangle):
    fn = "../../{}-{:.4}-FullTestNew/{}-{:.4}-FullTestNew".format(ms,mixangle,ms,mixangle)
    fn_df = fn + '/' + "df.mp4"
    
    if not os.path.isfile(fn+'/movie_arrays.npz'):
        print("Need to create data arrays.  Will take a few minutes.")
        create_movie_arrays(ms,mixangle)
    arrs = np.load(fn+'/movie_arrays.npz', allow_pickle=True)
    ts = sf.ts(ms,mixangle)*hbar
    
    def init():
        d_rate.set_data([], [])
        c_ratep.set_data([], [])
        c_ratem.set_data([], [])
        return d_rate, c_ratep, c_ratem
    
    def animate(i):
        e_vec = arrs['e'][i][:]
        d_vec = arrs['d'][i][:]
        c_vec = arrs['c'][i][:]
        T = arrs['T'][i]
        t = arrs['t'][i]
        a = arrs['a'][i]
        len_e = len(np.where(e_vec > 0)[0])+1

        eps = e_vec[:len_e]
        d = d_vec[:len_e]
        c = c_vec[:len_e]
        
        d_rate.set_data(eps,d*a)
        c_ratep.set_data(eps,c*a)
        c_ratem.set_data(eps,-c*a)
        time_text.set_text(r"$T = $"+"{:8.3} MeV".format(T) + '\n' + r"$T_{\rm cm} = $"+"{:8.3} MeV".format(1/a) + '\n' + "t = {:10.3} s".format(t * hbar))
        
        return d_rate, c_ratep, c_ratem, time_text
    
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes([0.2,0.12,0.7,0.85])
    d_rate, = ax.loglog([], [], linewidth=2, color='k')
    c_ratep, = ax.loglog([], [], linewidth=2, color='k', linestyle='--')
    c_ratem, = ax.loglog([], [], linewidth=2, color='r', linestyle='--')
    time_text = ax.text(250,50,"",fontsize=12)
    ax.set_xlim([0.5,3000])
    ax.set_ylim([1e-10,1e4])
    ax.set_yticks([1e-10,1e-5,1e0])
    ax.set_xlabel(r"$\epsilon = E_\nu / T_{\rm cm}$",fontsize=20)
    ax.set_ylabel(r"Rate, $(df/dt) / H$",fontsize=20)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = len(arrs['a']), interval = 33, blit=True)
    anim.save(fn_df)

def movie_TTcm(ms,mixangle):
    fn = "../../{}-{:.4}-FullTestNew/{}-{:.4}-FullTestNew".format(ms,mixangle,ms,mixangle)
    fn_TTcm = fn + '/' + "TTcm.mp4"
    
    if not os.path.isfile(fn+'/movie_arrays.npz'):
        print("Need to create movie arrays.  Will take a few minutes.")
        create_movie_arrays(ms,mixangle)
    arrs = np.load(fn+'/movie_arrays.npz', allow_pickle=True)
    ts = sf.ts(ms,mixangle)*hbar
    eq_Tcm = cs_Tcm(arrs['T']) 
    
    def init():
        t_rat.set_data([],[])
        t_rat_eq.set_data([],[])
        return t_rat, t_rat_eq
    
    def animate(i):
        T = arrs['T'][i]
        t = arrs['t'][i]
        a = arrs['a'][i]
        a_vec = arrs['a'][:(i+1)]
        T_vec = arrs['T'][:(i+1)]

        t_rat.set_data(T_vec,T_vec * a_vec)
        t_rat_eq.set_data(T_vec, T_vec / eq_Tcm[:(i+1)])
        time_text.set_text(r"$T = $"+"{:8.3} MeV".format(T) + '\n' + r"$T_{\rm cm} = $"+"{:8.3} MeV".format(1/a) + '\n' + "t = {:10.3} s".format(t * hbar))
        
        return t_rat, t_rat_eq, time_text
    
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes([0.2,0.12,0.7,0.85])
    t_rat, = ax.semilogx([],[],color='b')
    t_rat_eq, = ax.semilogx([],[],linestyle='--',color='k')
    time_text = ax.text(0.1,2.6,"",fontsize=12)
    ax.set_xlim(15,0.015)
    ax.set_ylim(0.95,3)
    ax.set_xlabel(r"$T$ (MeV)",fontsize=20)
    ax.set_ylabel(r"$T / T_{\rm cm}$",fontsize=20)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = len(arrs['a']), interval = 33, blit=True)
    anim.save(fn_TTcm)
        
def movie_n2p(ms,mixangle):
    fn = "../../{}-{:.4}-FullTestNew/{}-{:.4}-FullTestNew".format(ms,mixangle,ms,mixangle)
    fn_n2p = fn + '/' + "n2p.mp4"
    
    if not os.path.isfile(fn+'/movie_arrays.npz'):
        print("Need to create movie arrays.  Will take a few minutes.")
        create_movie_arrays(ms,mixangle)
    arrs = np.load(fn+'/movie_arrays.npz', allow_pickle=True) 
    
    def init():
        np_plot.set_data([],[])
        np_SM_plot.set_data([],[])
        pn_plot.set_data([],[])
        pn_SM_plot.set_data([],[])
        return np_plot, pn_plot, np_SM_plot, pn_SM_plot
    
    def animate(i):
        T_vec = arrs['T'][:(i+1)]
        Hub = arrs['Hub'][:(i+1)]
        Hub_SM = arrs['Hub_SM'][:(i+1)]
        n2p = arrs['np'][:(i+1)] / Hub
        n2p_SM = arrs['np_SM'][:(i+1)] / Hub_SM
        p2n = arrs['pn'][:(i+1)] / Hub
        p2n_SM = arrs['pn_SM'][:(i+1)] / Hub_SM
        
        np_plot.set_data(T_vec, n2p)
        np_SM_plot.set_data(eq_T_arr[:(i+1)], n2p_SM)
        pn_plot.set_data(T_vec, p2n)
        pn_SM_plot.set_data(eq_T_arr[:(i+1)], p2n_SM)
        
        return np_plot, pn_plot, np_SM_plot, pn_SM_plot

    fig = plt.figure(figsize=(10,6))
    ax = plt.axes([0.2,0.12,0.70,0.85])
    np_plot, = ax.loglog([],[],color='salmon',linewidth=2,label=r"$\lambda_{n \rightarrow p}$")
    np_SM_plot, = ax.loglog([],[],color='salmon',linewidth=1,linestyle='--')  
    pn_plot, = ax.loglog([],[],color='rebeccapurple',linewidth=2,label=r"$\lambda_{p \rightarrow n}$")
    pn_SM_plot, = ax.loglog([],[],color='rebeccapurple',linewidth=1,linestyle='--')
    time_text = ax.text(0.1,1000,"",fontsize=12)
    ax.axhline(1,color='0.50',linestyle='-.')
    ax.set_xlim(15,0.015)
    ax.set_ylim(1e-5,1e5)
    ax.set_yticks([1e-3,1e0,1e3])
    ax.set_xlabel(r"$T$ (MeV)",fontsize=20)
    ax.set_ylabel(r"$\lambda / H$",fontsize=20)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    #ax.text(10,4000,label=r"$\lambda_{n \rightarrow p}$",color='purple',fontsize=14)
    #ax.text(10,10000,label=r"$\lambda_{p \rightarrow n}$",color='gold',fontsize=14)
    ax.legend(loc="lower left",fontsize=14)
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = len(arrs['a']), interval = 33, blit=True)
    anim.save(fn_n2p)

def make_movies(ms,mixangle):
    movie_f(ms,mixangle)
    movie_df(ms,mixangle)
    movie_TTcm(ms,mixangle)
    movie_n2p(ms,mixangle)

    
#In[7]:
    
    
def make_movies_3(ms,mixangle):
    fn = "../../{}-{:.4}-FullTestNew/{}-{:.4}-FullTestNew".format(ms,mixangle,ms,mixangle)
    mv_fn = "decay_collision.mp4"
    movie_file_name = fn + '/' + mv_fn
    
    if not os.path.isfile(fn+'/movie_arrays.npz'):
        print("Need to create data arrays.  Will take a few minutes.")
        create_movie_arrays(ms,mixangle)
    arrs = np.load(fn+'/movie_arrays.npz', allow_pickle=True)
    ts = sf.ts(ms,mixangle)*hbar
    
    fig = plt.figure(figsize=(10,8))
    ax1 = plt.axes([0.09,0.1,0.415,0.42])
    ax0 = plt.axes([0.09,0.52,0.415,0.42])
    ax2 = plt.axes([0.65,0.25,0.3,0.5])
    nu, = ax0.loglog([], [], linewidth=3, color='b')
    ax0.set_ylim([1e-10,10])
    ax0.set_yticks([1e-9,1e-6,1e-3,1e0])
    ax0.set_xlim([0.5,3000])
    ax1.set_xlim([0.5,3000])
    ax0.set_ylabel(r"$\epsilon^3 \, f(\epsilon)$")
    ax0.set_title("{} MeV, {:.3} s decay lifetime".format(ms,ts))
    ax0.xaxis.set_major_formatter(plt.NullFormatter())
    
    d_rate, = ax1.loglog([], [], linewidth=2, color='k')
    c_ratep, = ax1.loglog([], [], linewidth=2, color='k', linestyle='--')
    c_ratem, = ax1.loglog([], [], linewidth=2, color='r', linestyle='--')
    ax1.set_ylim([1e-25,1e4])
    ax1.set_yticks([1e-25,1e-20,1e-15,1e-10,1e-5,1e0])
    ax1.set_xlabel(r"$\epsilon = E_\nu / T_{\rm cm}$")
    ax1.set_ylabel(r"Rate ($df/da$)")

    ax2.set_xlim(15,.015)
    ax2.set_ylim(.0015,15)
    ax2.set_ylabel(r"$T$ (MeV)")
    ax2.set_xlabel(r"$T_{cm}$ (MeV)")
    temp_temp, = ax2.loglog([], [], linewidth=2, color = 'k')
    diag, = ax2.loglog([], [], color='0.75', linestyle=':')
    time_text = ax2.text(10,3e-3,"",fontsize=12)
    
    def init():
        nu.set_data([], [])
        d_rate.set_data([], [])
        c_ratep.set_data([], [])
        c_ratem.set_data([], [])
        temp_temp.set_data([],[])
        diag.set_data([],[])
        time_text.set_text("")
        return nu, d_rate, c_ratep, c_ratem, time_text, temp_temp, diag
    
    def animate(i):
        e_vec = arrs['e'][i][:]
        f_vec = arrs['fe'][i][:]
        d_vec = arrs['d'][i][:]
        c_vec = arrs['c'][i][:]
        T = arrs['T'][i]
        t = arrs['t'][i]
        a = arrs['a'][i]
        a_vec = arrs['a'][:(i+1)]
        T_vec = arrs['T'][:(i+1)]
        len_e = len(np.where(e_vec > 0)[0])+1
        
        eps = e_vec[:len_e]
        fv = f_vec[:len_e]
        d = d_vec[:len_e]
        c = c_vec[:len_e]
        
        nu.set_data(eps, eps**3 * fv / (2 * np.pi**3))
        d_rate.set_data(eps,d)
        c_ratep.set_data(eps,c)
        c_ratem.set_data(eps,-c)
        time_text.set_text(r"$T = $"+"{:8.3} MeV".format(T) + '\n\n' + "$T_{cm} = $"+"{:8.3} MeV".format(1/a) + '\n' + "t = {:10.3} s".format(t * hbar))
        temp_temp.set_data(1/a_vec,T_vec)
        diag.set_data(np.linspace(0.001,20),np.linspace(0.001,20))
        
        return nu, d_rate, c_ratep, c_ratem, time_text, temp_temp, diag
    
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = len(arrs['a']), interval = 1, blit=True)
    
    anim.save(movie_file_name)
