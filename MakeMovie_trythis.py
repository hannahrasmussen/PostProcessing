#!/usr/bin/env python
# coding: utf-8

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from matplotlib import animation
import matplotlib.pyplot as plt
import os #what's this?
import scipy.interpolate as sp

import movie_support_functions as mv #in the same package, for now?

from nu_nu_coll import nu_nu_collisions as coll
from BasicCode import basic_code as trunc
from CollisionApprox import new_Collision_approx as ca
from N2P import N2P_ratio_not_eq as nprate #?? I call it N2P elsewhere


# In[2]:


def make_file_list(folder_name,max_files=200,Tmin = 0.0001):
    output = []
    for i in range(max_files):
        test_name = folder_name + '/full-' + str(i)
        try:
            y_g = np.load(test_name + "-y.npy")
            if (y_g[-2][0] > Tmin):
                output.append(test_name)
            else:
                break
        except:
            break
    for i in range(max_files):
        test_name = folder_name + '/full_cont_new-' + str(i)
        try:
            y_g = np.load(test_name + "-y.npy")
            if (y_g[-2][0] > Tmin):
                output.append(test_name)
            else:
                break
        except:
            break
    return output

def make_aye(run_file):
    a_g = np.load(run_file + "-a.npy")
    y_g = np.load(run_file + "-y.npy")
    e_g = np.load(run_file + "-e.npy")
    
    y_vals = np.zeros((len(a_g),len(y_g)-3))
    t_vals = np.zeros(len(a_g))
    T_vals = np.zeros(len(a_g))
    
    for i in range(len(a_g)):
        t_vals[i] = y_g[-1][i]
        T_vals[i] = y_g[-2][i]
        for j in range(len(y_g)-3):
            y_vals[i][j] = y_g[j][i]
    
    return a_g, y_vals, e_g, T_vals, t_vals, y_g


# In[14]:

def f_nu(E,Tcm):
    return 1/(np.exp(E/Tcm)+1)

def create_data_arrays(ms,mixangle,rates=False):
    folder = "..../{}-{:.4}-FullTestNew/{}-{:.4}-FullTestNew".format(ms,mixangle,ms,mixangle)
    files = make_file_list(folder)
    
    a = []
    y = []
    e = []
    T = []
    t = []
    ym =[]
    Num_steps = 1
    max_y_dim = 1
    for i in range(len(files)):
        av,yv,ev,Tv,tv,ymv = make_aye(files[i])
        a.append(av)
        y.append(yv)
        e.append(ev)
        T.append(Tv)
        t.append(tv)
        ym.append(ymv)
        Num_steps += len(av) -1
        max_y_dim = max(max_y_dim, len(yv[0]))
    a_tot = np.zeros(Num_steps)
    y_tot = np.zeros((Num_steps,max_y_dim))
    e_tot = np.zeros((Num_steps,max_y_dim))
    T_tot = np.zeros(Num_steps)
    t_tot = np.zeros(Num_steps)
    y_mat = np.zeros((max_y_dim+3,Num_steps))
    
    a_tot[0] = a[0][0]
    T_tot[0] = T[0][0]
    t_tot[0] = t[0][0]

    for i in range(len(y[0][0])):
        y_tot[0][i] = y[0][0][i]
        y_mat[i][0] = y[0][0][i]
        e_tot[0][i] = e[0][i]
        
    y_mat[-1][0] = t_tot[0]
    y_mat[-2][0] = T_tot[0]

    k = 1
    for i in range(len(files)):
        print(files[i])
        for j in range(1,len(a[i])):
            a_tot[k] = a[i][j]
            T_tot[k] = T[i][j]
            t_tot[k] = t[i][j]
            y_mat[-1][k] = t_tot[k]
            y_mat[-2][k] = T_tot[k]

            for ii in range(len(e[i])):
                e_tot[k][ii] = e[i][ii]

            for ii in range(len(y[i][j])):
                y_tot[k][ii] = y[i][j][ii]
                y_mat[ii][k] = y[i][j][ii]
            k += 1
            
    if rates:
        eq_a_array = np.load("../eq_arrs/eq_arrs/Luke_a.npy")
        eq_T_array = np.load("../eq_arrs/eq_arrs/Luke_T.npy")
        eq_eta_array = np.load("../eq_arrs/eq_arrs/Luke_eta.npy")
        eq_e_array = np.linspace(0,200,201)
        n2p_st = np.zeros(len(eq_a_array))
        p2n_st = np.zeros(len(eq_a_array))
        Hub_st = np.zeros(len(eq_a_array))
        
        d_tot = np.zeros((Num_steps,max_y_dim+3))
        c_tot = np.zeros((Num_steps,max_y_dim))
        n2p = np.zeros(Num_steps)
        p2n = np.zeros(Num_steps)
        Hub = np.zeros(Num_steps)
        
        cs_eta = sp.CubicSpline(np.flip(eq_T_array),np.flip(eq_eta_array))
        cs_Tcm = sp.CubicSpline(np.flip(eq_T_array),np.flip(1/eq_a_array))
        
        eq_Tcm = cs_Tcm(T_tot)
        eq_eta = cs_eta(T_tot)
        for i in range(Num_steps):
            len_e = len(np.where(e_tot[i] > 0)[0])+1
            n2p[i] = nprate.varlam_np(a_tot[i], T_tot[i], y_tot[i][:len_e], e_tot[i][:len_e], eq_eta[i])
            p2n[i] = nprate.varlam_pn(a_tot[i], T_tot[i], y_tot[i][:len_e], e_tot[i][:len_e], eq_eta[i])
        for i in range(len(eq_a_array)):
            n2p_st[i] = nprate.varlam_np(eq_a_array[i], eq_T_array[i], f_nu(eq_e_array/eq_a_array[i],1/eq_a_array[i]), eq_e_array, eq_eta_array[i])
            p2n_st[i] = nprate.varlam_pn(eq_a_array[i], eq_T_array[i], f_nu(eq_e_array/eq_a_array[i],1/eq_a_array[i]), eq_e_array, eq_eta_array[i])
    
        k = 1
        for i in range(len(files)):
            if k == 1:
                len_e = len(np.where(e_tot[0] > 0)[0])+1
                A_model, n_model = ca.model_An(a_tot[0], T_tot[0], 1/(0.9 * a_tot[0] * T_tot[0]))
                fn = mv.make_collision_fn(ms,mixangle,e_tot[0][:len_e],A_model,n_model)
            else:
                if(a_tot[k]>4000): #an error happens around a=4200 in the collision code for some reason
                    break
                len_e = len(np.where(e_tot[k] > 0)[0])+1
                A_model, n_model = ca.model_An(a_tot[k], T_tot[k], 1/(0.9 * a_tot[k] * T_tot[k]))
                fn = mv.make_collision_fn(ms,mixangle,e_tot[k][:len_e],A_model,n_model)
                
            print(T_tot[k-1], 1/a_tot[k-1], A_model, n_model)
            for j in range(1,len(a[i])):
                y_temp = np.zeros(len_e+3)
                y_temp[:len_e] = y_tot[k,:len_e]
                y_temp[-2] = T_tot[k]
                y_temp[-1] = t_tot[k]
                d, c = fn(a_tot[k],y_temp)
                d_tot[k,:len_e+3] = d
                c_tot[k,:len_e] = c
                Hub[k] = (d[-1] * a_tot[k])**(-1)
                
                k += 1
                
        Hub[0] = Hub[1] 
        
        for i in range(1,len(eq_a_array)):
            dtda = mv.time_derivative(eq_a_array[i],eq_T_array[i])
            Hub_st[i] = (dtda * eq_a_array[i])**(-1)
        Hub_st[0] = Hub_st[1]
        
        np.savez(folder+'/movie-arrays',a=a_tot, y=y_tot, e=e_tot, T=T_tot, t=t_tot, ym=y_mat, d=d_tot, c=c_tot, np=n2p, np_SM=n2p_st, pn=p2n, pn_SM=p2n_st, Hub=Hub, Hub_st=Hub_st)
        return a_tot, y_tot, e_tot, T_tot, t_tot, y_mat, d_tot, c_tot, n2p, n2p_st, p2n, p2n_st, Hub, Hub_st
    else:
        return a_tot, y_tot, e_tot, T_tot, t_tot, y_mat

#def nuc(a, T, f, e):
    

# In[23]:


def make_video(mH,mixangle):
    fn = "..../{}-{:.4}-FullTestNew/{}-{:.4}-FullTestNew".format(mH,mixangle,mH,mixangle)
    movie_file_name0 = fn + '/' + "f.mp4"
    movie_file_name1 = fn + '/' + "df.mp4"
    movie_file_name2 = fn + '/' + "TTcm.mp4"
    movie_file_name3 = fn + '/' + "n2p.mp4"
    movie_file_name4 = fn + '/' + "n2p_SM.mp4"
    
    if not os.path.isfile(fn+'/movie-arrays.npz'):
        print("Need to create data arrays.  Will take a few minutes.")
        create_data_arrays(mH,mixangle,True)
    arrs = np.load(fn+'/movie-arrays.npz')
    
    hbar = 6.58212e-22
    tH = trunc.tH(mH,mixangle)*hbar
    
    eq_a_array = np.load("../eq_arrs/eq_arrs/Luke_a.npy")
    eq_T_array = np.load("../eq_arrs/eq_arrs/Luke_T.npy")
    eq_eta_array = np.load("../eq_arrs/eq_arrs/Luke_eta.npy")
    
    cs_eta = sp.CubicSpline(np.flip(eq_T_array),np.flip(eq_eta_array))
    cs_Tcm = sp.CubicSpline(np.flip(eq_T_array),np.flip(1/eq_a_array))
    
    eq_Tcm = cs_Tcm(arrs['T']) 
    
    def init0():
        e_thermal.set_data([], [])
        nu.set_data([], [])
        return e_thermal, nu
    
    def init1():
        d_rate.set_data([], [])
        c_ratep.set_data([], [])
        c_ratem.set_data([], [])
        return d_rate, c_ratep, c_ratem
    
    def init2():
        t_rat.set_data([],[])
        t_rat_eq.set_data([],[])
        return t_rat, t_rat_eq
    
    def init3():
        pn_plot.set_data([],[])
        np_plot.set_data([],[])
        return pn_plot, np_plot
    
    def init4():
        pn_SM_plot.set_data([],[])
        np_SM_plot.set_data([],[])
        return pn_SM_plot, np_SM_plot

    
    def animate0(i):
        e_vec = arrs['e'][i,:]
        y_vec = arrs['y'][i,:]
        T = arrs['T'][i]
        t = arrs['t'][i]
        a = arrs['a'][i]
        len_e = len(np.where(e_vec > 0)[0])+1

        eps = e_vec[:len_e]
        yv = y_vec[:len_e]

        e_thermal.set_data(eps, eps**3 / (np.exp(eps / a /  eq_Tcm[i]) +1) / (2 * np.pi**3))
        nu.set_data(eps, eps**3 * yv / (2 * np.pi**3))
        time_text.set_text(r"$T = $"+"{:8.3} MeV".format(T) + '\n' + r"$T_{\rm cm} = $"+"{:8.3} MeV".format(1/a) + '\n' + "t = {:10.3} s".format(t * hbar))
        
        return e_thermal, nu, time_text
    
    def animate1(i):
        e_vec = arrs['e'][i,:]
        d_vec = arrs['d'][i,:]
        c_vec = arrs['c'][i,:]
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
    
    def animate2(i):
        T = arrs['T'][i]
        t = arrs['t'][i]
        a = arrs['a'][i]
        a_vec = arrs['a'][:(i+1)]
        T_vec = arrs['T'][:(i+1)]

        t_rat.set_data(T_vec,T_vec * a_vec)
        t_rat_eq.set_data(T_vec, T_vec / eq_Tcm[:(i+1)])
        time_text.set_text(r"$T = $"+"{:8.3} MeV".format(T) + '\n' + r"$T_{\rm cm} = $"+"{:8.3} MeV".format(1/a) + '\n' + "t = {:10.3} s".format(t * hbar))
        
        return t_rat, t_rat_eq, time_text
    
    def animate3(i):
        T = arrs['T'][i]
        t = arrs['t'][i]
        a = arrs['a'][i]
        T_vec = arrs['T'][:(i+1)]
        Hub = arrs['Hub'][:(i+1)]
        n2p = arrs['np'][:(i+1)] / Hub
        p2n = arrs['pn'][:(i+1)] / Hub

        pn_plot.set_data(T_vec, p2n)
        np_plot.set_data(T_vec, n2p)
        time_text.set_text(r"$T = $"+"{:8.3} MeV".format(T) + '\n' + r"$T_{\rm cm} = $"+"{:8.3} MeV".format(1/a) + '\n' + "t = {:10.3} s".format(t * hbar))
        
        return pn_plot, np_plot, time_text
        
    def animate4(i):
        T_vec = eq_T_array[:(i+1)]
        Hub_SM = arrs['Hub_st'][:(i+1)] 
        n2p_SM = arrs['np_SM'][:(i+1)] / Hub_SM
        p2n_SM = arrs['pn_SM'][:(i+1)] / Hub_SM

        pn_SM_plot.set_data(T_vec, p2n_SM)
        np_SM_plot.set_data(T_vec, n2p_SM)

        return pn_SM_plot, np_SM_plot
    
    fig0 = plt.figure(figsize=(10,6)) #epsilon vs adjusted occupation fraction
    ax0 = plt.axes([0.2,0.12,0.7,0.85])
    e_thermal, = ax0.loglog([], [], linewidth=2, linestyle='--',color='b')
    nu, = ax0.loglog([], [], linewidth=2, color='k')
    time_text = ax0.text(250,0.1,"",fontsize=12)
    ax0.set_ylim([1e-10,10])
    ax0.set_yticks([1e-9,1e-6,1e-3,1e0])
    ax0.set_xlim([0.5,3000])
    ax0.set_xticks([1e0,1e1,1e2,1e3])
    ax0.set_xlabel(r"$\epsilon = E_\nu / T_{\rm cm}$",fontsize=20)
    ax0.set_ylabel(r"$\epsilon^3 \, f(\epsilon)$",fontsize=20)
    #ax0.set_title("{} MeV, {:.3} s decay lifetime".format(mH,tH))
    ax0.tick_params(axis="x", labelsize=14)
    ax0.tick_params(axis="y", labelsize=14)
    anim0 = animation.FuncAnimation(fig0, animate0, init_func = init0, frames = len(arrs['a']), interval = 33, blit=True)
    anim0.save(movie_file_name0)

    fig1 = plt.figure(figsize=(10,6)) #epsilon vs df/dt
    ax1 = plt.axes([0.2,0.12,0.7,0.85])
    d_rate, = ax1.loglog([], [], linewidth=2, color='k')
    c_ratep, = ax1.loglog([], [], linewidth=2, color='k', linestyle='--')
    c_ratem, = ax1.loglog([], [], linewidth=2, color='r', linestyle='--')
    time_text = ax1.text(250,50,"",fontsize=12)
    ax1.set_xlim([0.5,3000])
    ax1.set_ylim([1e-10,1e4])
    ax1.set_yticks([1e-10,1e-5,1e0])
    ax1.set_xlabel(r"$\epsilon = E_\nu / T_{\rm cm}$",fontsize=20)
    ax1.set_ylabel(r"Rate, $(df/dt) / H$",fontsize=20)
    ax1.tick_params(axis="x", labelsize=14)
    ax1.tick_params(axis="y", labelsize=14)
    anim1 = animation.FuncAnimation(fig1, animate1, init_func = init1, frames = len(arrs['a']), interval = 33, blit=True)
    anim1.save(movie_file_name1)
    
    fig2 = plt.figure(figsize=(10,6)) #T/Tcm
    ax2 = plt.axes([0.2,0.12,0.7,0.85])
    t_rat, = ax2.semilogx([],[],color='b')
    t_rat_eq, = ax2.semilogx([],[],linestyle='--',color='k')
    time_text = ax2.text(0.1,2.6,"",fontsize=12)
    ax2.set_xlim(15,0.015)
    ax2.set_ylim(0.95,3)
    ax2.set_xlabel(r"$T$ (MeV)",fontsize=20)
    ax2.set_ylabel(r"$T / T_{\rm cm}$",fontsize=20)
    ax2.tick_params(axis="x", labelsize=14)
    ax2.tick_params(axis="y", labelsize=14)
    anim2 = animation.FuncAnimation(fig2, animate2, init_func = init2, frames = len(arrs['a']), interval = 33, blit=True)
    anim2.save(movie_file_name2)

    fig3 = plt.figure(figsize=(10,6)) #proton to neutron graph
    ax3 = plt.axes([0.2,0.12,0.70,0.85])
    pn_plot, = ax3.loglog([],[],color='rebeccapurple',linewidth=2,label=r"$\lambda_{p \rightarrow n}$")
    np_plot, = ax3.loglog([],[],color='salmon',linewidth=2,label=r"$\lambda_{n \rightarrow p}$")
    time_text = ax3.text(0.1,1000,"",fontsize=12)
    ax3.axhline(1,color='0.50',linestyle='-.')
    ax3.set_xlim(15,0.015)
    ax3.set_ylim(1e-5,1e5)
    ax3.set_yticks([1e-3,1e0,1e3])
    ax3.set_xlabel(r"$T$ (MeV)",fontsize=20)
    ax3.set_ylabel(r"$\lambda / H$",fontsize=20)
    ax3.tick_params(axis="x", labelsize=14)
    ax3.tick_params(axis="y", labelsize=14)
    ax3.legend(loc="lower left",fontsize=14)
    anim3 = animation.FuncAnimation(fig3, animate3, init_func = init3, frames = len(arrs['a']), interval = 33, blit=True)
    anim3.save(movie_file_name3)
    
    fig4 = plt.figure(figsize=(10,6)) #standard model proton to neutron graph
    ax4 = plt.axes([0.2,0.12,0.70,0.85])
    pn_SM_plot, = ax4.loglog([],[],color='rebeccapurple',linewidth=1,linestyle='--',label=r"$\lambda_{p \rightarrow n}$")
    np_SM_plot, = ax4.loglog([],[],color='salmon',linewidth=1,linestyle='--',label=r"$\lambda_{n \rightarrow p}$")  
    time_text = ax3.text(0.1,1000,"",fontsize=12)
    ax4.axhline(1,color='0.50',linestyle='-.')
    ax4.set_xlim(15,0.015)
    ax4.set_ylim(1e-5,1e5)
    ax4.set_yticks([1e-3,1e0,1e3])
    ax4.set_xlabel(r"$T$ (MeV)",fontsize=20)
    ax4.set_ylabel(r"$\lambda / H$",fontsize=20)
    ax4.tick_params(axis="x", labelsize=14)
    ax4.tick_params(axis="y", labelsize=14)
    #ax3.text(10,4000,label=r"$\lambda_{n \rightarrow p}$",color='purple',fontsize=14)
    #ax3.text(10,10000,label=r"$\lambda_{p \rightarrow n}$",color='gold',fontsize=14)
    ax4.legend(loc="lower left",fontsize=14)
    anim4 = animation.FuncAnimation(fig4, animate4, init_func = init4, frames = len(arrs['a']), interval = 33, blit=True)
    anim4.save(movie_file_name4)


def make_video_old(mH,mixangle,mv_fn="decay_collision.mp4"):
    fn = "..../{}-{:.4}-FullTestNew/{}-{:.4}-FullTestNew".format(mH,mixangle,mH,mixangle)
    movie_file_name = fn + '/' + mv_fn
    
    if not os.path.isfile(fn+'/movie-arrays.npz'):
        print("Need to create data arrays.  Will take a few minutes.")
        create_data_arrays(mH,mixangle,True)
    arrs = np.load(fn+'/movie-arrays.npz')
    
    hbar = 6.58212e-22
    tH = trunc.tH(mH,mixangle)*hbar
    

    fig = plt.figure(figsize=(10,8))
    ax1 = plt.axes([0.09,0.1,0.415,0.42])
    ax0 = plt.axes([0.09,0.52,0.415,0.42])
    ax2 = plt.axes([0.65,0.25,0.3,0.5])
#    ics, = ax0.loglog([], [], linewidth=2, linestyle='--',color='0.50')
#    e_thermal, = ax0.loglog([], [], linewidth=2, linestyle='--',color='0.75')
    nu, = ax0.loglog([], [], linewidth=3, color='b')
    ax0.set_ylim([1e-10,10])
    ax0.set_yticks([1e-9,1e-6,1e-3,1e0])
    ax0.set_xlim([0.5,3000])
    ax1.set_xlim([0.5,3000])
    ax0.set_ylabel(r"$\epsilon^3 \, f(\epsilon)$")
    ax0.set_title("{} MeV, {:.3} s decay lifetime".format(mH,tH))
    ax0.xaxis.set_major_formatter(plt.NullFormatter())
    
    d_rate, = ax1.loglog([], [], linewidth=2, color='k')
    c_ratep, = ax1.loglog([], [], linewidth=2, color='k', linestyle='--')
    c_ratem, = ax1.loglog([], [], linewidth=2, color='r', linestyle='--')
#    cel_ratep, = ax1.loglog([], [], linewidth=2, color='k', linestyle=':')
#    cel_ratem, = ax1.loglog([], [], linewidth=2, color='r', linestyle=':')
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
#        ics.set_data([], [])
#        e_thermal.set_data([], [])
        nu.set_data([], [])
        d_rate.set_data([], [])
        c_ratep.set_data([], [])
        c_ratem.set_data([], [])
#        cel_ratep.set_data([],[])
#        cel_ratem.set_data([],[])
        temp_temp.set_data([],[])
        diag.set_data([],[])
        time_text.set_text("")
        return nu, d_rate, c_ratep, c_ratem, time_text, temp_temp, diag
#        return ics, e_thermal, nu, d_rate, c_ratep, c_ratem, cel_ratep, cel_ratem, time_text, temp_temp, diag
    
    def animate(i):
        e_vec = arrs['e'][i,:]
        y_vec = arrs['y'][i,:]
        d_vec = arrs['d'][i,:]
        c_vec = arrs['c'][i,:]
        T = arrs['T'][i]
        t = arrs['t'][i]
        a = arrs['a'][i]
        a_vec = arrs['a'][:(i+1)]
        T_vec = arrs['T'][:(i+1)]
        len_e = len(np.where(e_vec > 0)[0])+1

        eps = e_vec[:len_e]
        yv = y_vec[:len_e]
        d = d_vec[:len_e]
        c = c_vec[:len_e]

#        ics.set_data(eps,eps**3 / (np.exp(eps) + 1) / (2 * np.pi**3))
#        e_thermal.set_data(eps, eps**3 / (np.exp(eps / T[i] / a[i]) +1) / (2 * np.pi**3))
        nu.set_data(eps, eps**3 * yv / (2 * np.pi**3))
        d_rate.set_data(eps,d)
        c_ratep.set_data(eps,c)
        c_ratem.set_data(eps,-c)
#        cel_ratep.set_data(e[i][:-1],cel[i][:-1])
#        cel_ratem.set_data(e[i][:-1],-cel[i][:-1])
        time_text.set_text(r"$T = $"+"{:8.3} MeV".format(T) + '\n\n' + "$T_{cm} = $"+"{:8.3} MeV".format(1/a) + '\n' + "t = {:10.3} s".format(t * hbar))
        temp_temp.set_data(1/a_vec,T_vec)
        diag.set_data(np.linspace(0.001,20),np.linspace(0.001,20))
        
        return nu, d_rate, c_ratep, c_ratem, time_text, temp_temp, diag
#        return ics, e_thermal, nu, d_rate, c_ratep, c_ratem, cel_ratep, cel_ratem, time_text, temp_temp, diag
    
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = len(arrs['a']), interval = 1, blit=True)
    
    anim.save(movie_file_name)

