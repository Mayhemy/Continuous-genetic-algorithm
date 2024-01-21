# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random
import sys
import math
from locale import atof

import re
import numpy as np
from random import random as rnd
import subprocess
from subprocess import PIPE, Popen
from random import gauss, randrange

from matplotlib import pyplot as plot

x_r = [
    0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
    1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90,
    2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90,
    3.00, 3.10, 3.20, 3.30, 3.40, 3.50, 3.60, 3.70, 3.80, 3.90,
    4.00, 4.10, 4.20, 4.30, 4.40, 4.50, 4.60, 4.70, 4.80, 4.90
]

y_k = [
    7.0000, 7.4942, 8.1770, 9.0482, 10.1080, 11.3562, 12.7929, 14.4181, 16.2319, 18.2341,
    20.4248, 22.8040, 25.3717, 28.1279, 31.0726, 34.2058, 37.5274, 41.0376, 44.7363, 48.6234,
    52.6991, 56.9633, 61.4159, 66.0571, 70.8867, 75.9049, 81.1115, 86.5066, 92.0903, 97.8624,
    103.8230, 109.9721, 116.3097, 122.8358, 129.5504, 136.4535, 143.5451, 150.8252, 158.2938, 165.9509,
    173.7964, 181.8305, 190.0531, 198.4641, 207.0637, 215.8518, 224.8283, 233.9933, 243.3469, 252.8889
]


def calc_neuron(neurons, input_weights, input_values, bias):
    suma = bias
    for x in range(neurons):
        suma += input_weights[x] * input_values[x]
    return suma


L1N = 4
L2N = 4

layer1 = [[float for y in range(1)] for x in range(L1N)]
layer2 = [[float for y in range(L1N)] for x in range(L2N)]
layer3 = [[float for y in range(L2N)] for x in range(1)]

bias1 = [float for x in range(L1N)]
bias2 = [float for x in range(L2N)]
bias3 = [float for x in range(1)]

output0 = [float for x in range(1)]
output1 = [float for x in range(L1N)]
output2 = [float for x in range(L2N)]

# def pomocna(argv1):
#     return argv1[0]*math.sin(4*argv1[0])+1.1*argv1[1]*math.sin(2*argv1[1])

def pomocna(argv1):
    argv1 = [str(x) for x in argv1]
    argglupost = ["stasta"]
    argv = argglupost+argv1
    # print(argv)
    y_r = y_k
    total_expected_args = 1 + L1N + L1N * L2N + L2N + L1N + L2N + 1
    if len(argv) != total_expected_args:
        print("Broj argumenata nije odgovarajuci, ocekivano %d realnih vrednosti.\n", total_expected_args - 1)
        exit(1)
    i = 0
    j = 0
    k = 0
    ai = 1
    # layer1
    for x in range(L1N):
        layer1[x][0] = atof(argv[ai])
        ai += 1
    # layer2
    for x in range(L2N):
        for y in range(L1N):
            layer2[x][y] = atof(argv[ai])
            ai += 1
    # layer3
    for x in range(L2N):
        layer3[0][x] = atof(argv[ai])
    # bias1
    for x in range(L1N):
        bias1[x] = atof(argv[ai])
        ai += 1
    # bias2
    for x in range(L2N):
        bias2[x] = atof(argv[ai])
        ai += 1
    # bias3
    bias3[0] = atof(argv[ai])
    # idemo dalje!
    mse = 0.0

    for k in range(50):
        output0[0] = x_r[k]
        # layer1
        for x in range(L1N):
            output1[x] = calc_neuron(1, layer1[x], output0, bias1[x])
        # layer2
        for x in range(L2N):
            output2[x] = calc_neuron(L1N, layer2[x], output1, bias2[x])
        # layer3
        val = calc_neuron(L2N, layer3[0], output2, bias3[0])
        err = pow(y_r[k]-val, 2)
        mse += err

    return round(mse/50, 5)


def individual():
    hromozom = [0]*number_of_genes
    index = 0
    for x in range(number_of_genes):
        hromozom[index] = round(np.random.uniform(lower_bound, upper_bound), 3)
        # round(rnd()*(upper_limit-lower_limit) +lower_limit,1)#np.random.uniform(-3,3)
        index += 1
    return hromozom


def population(number_of_individuals):
    niz = [0]*number_of_individuals
    index = 0
    for x in range(number_of_individuals):
        niz[index] = individual()
        index += 1
    return niz


def funkcija_troska(hromozom):
    path_and_args = ["GA-KXZK.exe"]
    hromozom = [str(x) for x in hromozom]
    path_and_args.extend(hromozom)
    output = subprocess.Popen(path_and_args, stdout=subprocess.PIPE)
    out, err = output.communicate()
    return float(out)


def mutiraj(hromozom):
    for i in range(len(hromozom)):
        hromozom[i] = round(np.random.uniform(lower_bound, upper_bound), 3)
        # round(rnd()*(upper_limit-lower_limit) +lower_limit,1)#np.random.uniform(-3,3)
    return hromozom


def selection(pop, vel):
    z = []
    while len(z) < vel:
        z.append(random.choice(pop))
    najbolji = None
    najbolji_f = None
    for e in z:
        ff = pomocna(e)
        if najbolji is None or ff < najbolji_f:
            najbolji_f = ff
            najbolji = e
    return najbolji

# def genetski():
#     tryout=[ -3.154 , 12.747,  -8.493 , -6.868  , 9.152  , 3.314 ,  3.847 , 11.643,  -7.948,
#  -15.881 , -9.835 ,-15.037 ,  6.631 , -5.417,  -4.54 ,  12.858 , -1.096 , -2.371,
#    8.831,  -6.725,  -0.176,  -5.473 ,  5.701  ,-3.37 ,  -8.133 ,-11.854 , -0.28,
#    2.232, -17.047 ,  5.187,  10.427,  -2.344 ,-10.201]
#     pop=tryout*50
#     tryout1=[1.072,  2.921, -5.63 ,  1.167 , 1.55  , 8.115 , 1.214,  2.661, -1.32,  -2.926,
#   2.001, -4.587, -3.141 , 2.717, -2.649 , 2.094 , 2.218 ,-6.995,  5.31,  -2.736,
#  -1.469, -6.938 , 4.765 ,-8.578 ,-7.811, -5.975, -2.434, -4.21 ,  5.743, -0.028,
#   8.01,   3.755 , 4.128]
#     pop=population(50)
#     h1=selection(pop,4)
#     print(pomocna(h1))
#     print(funkcija_troska(h1))
#     print(" ")
#     h2=selection(pop,4)
#     print(pomocna(h2))
#     print(funkcija_troska(h2))
#     if random.random()<mut_rate:


def heuristic_crossover(hromozom1, hromozom2, n):
    np_hromozom1 = np.array(hromozom1)
    np_hromozom2 = np.array(hromozom2)
    if pomocna(hromozom1) <= pomocna(hromozom2):
        for x in range(n):
            r = random.random()
            razlika_vektora = np_hromozom1 - np_hromozom2
            potomak = np.multiply(razlika_vektora, r)
            potomak = np.array(potomak) + np_hromozom1
            if validan(potomak):
                return np.round(potomak, 3)
    else:
        for x in range(n):
            r = random.random()
            razlika_vektora = np_hromozom2 - np_hromozom1
            potomak = np.multiply(razlika_vektora, r)
            potomak = np.array(potomak) + np_hromozom2
            if validan(potomak):
                return np.round(potomak, 3)
    return [100]

def validan(hromozom):
    for x in hromozom:
        if x>upper_bound or x<lower_bound:
            return False
    return True



def iscrtaj(best_lista, avg_lista, broj_generacije, pop_vel):
    boje = ["blue", "red", "yellow"]
    for i in range(len(best_lista)):
        x_list=list(range(broj_generacije[i]))
        y_list=best_lista[i]
        plot.plot(x_list,y_list,color=boje[i],label=str(i+1))
        plot.xlabel("Broj generacije")
        plot.ylabel("Najbolja vrednost funkcije")
        plot.title("Najbolja vrednost funkcije po generaciji")
    filename = "Najbolji_" + str(pop_vel) + '.pdf'
    plot.legend(loc='upper right')
    plot.savefig(filename)

    plot.clf()
    for i in range(len(avg_lista)):
        x_list = list(range(broj_generacije[i]))
        y_list = avg_lista[i]
        plot.plot(x_list, y_list, color=boje[i],label=str(i+1))
        plot.xlabel("Broj generacije")
        plot.ylabel("Srednja vrednost funkcije")
        plot.title("Srednja vrednost funkcije po generaciji")
    filename = "Srednji_" + str(pop_vel) + '.pdf'
    plot.legend(loc='upper right')
    plot.savefig(filename)

max_iter = 150
conv_numb = 100
#iz fajla
#testiralo se sa 53555645, za potrebe lepseg grafika testiram sa drugom vrednoscu 4325
default_values=[3, 10, 0.28, 0.8, 53555645, 'GA-KXZK.txt', -3, 3, 33, 2, 20]
broj_pokretanja=3
pop_vel = 10
npop_vel = 10
mut_rate = 0.28
uvodjenje_procenta_potomaka=1
rand_seed=53555645
random.seed(rand_seed)
np.random.seed(rand_seed)
out_path="GA-KXZK.txt"
lower_bound = -3
upper_bound = 3
number_of_genes = 33
velicina_turnira=2
velicina_heuristickog_ukrstanja=20
imena_promenljivih=["broj_pokretanja","pop_vel","mut_rate","uvodjenje_procenta_potomaka","rand_seed","path","lower_bound","upper_bound","number_of_genes","velicina_turnira","velicina_heuristickog_ukrstanja"]

def ucitavanje_iz_fajla(path):
    ulazni_niz = []
    ispis=[]
    lines = []
    try:
        with open(path,'r') as f:
            lines = f.readlines()
            for line in lines:
                pomocni_niz=line.split(",")
                for x in range(len(pomocni_niz)):
                    if not x==5:
                        ulazni_niz.append(int(pomocni_niz[x]))
                    else:
                        ulazni_niz.append(pomocni_niz[x])
    except:
        try:
            with open('cfg.txt','r') as f:
                lines = f.readlines()
            for number_of_line in range(len(lines)):
                split=(lines[number_of_line]).split(":",1)
                ispis.append(split[0])
                argument=split[1].rstrip()
                if not argument:
                    argument='0'
                if not number_of_line==5:
                    argument=int(argument)
                ulazni_niz.append(argument)
        except:
            ulazni_niz=default_values[:]
    print(ulazni_niz)
    print(default_values)
    for i in range(len(default_values)):
        if ulazni_niz[i]==0 or (i==5 and str(ulazni_niz[i]).isdigit()):
            ulazni_niz[i]=default_values[i]
    print("Niz sa kojim se radi:", ulazni_niz)
    global broj_pokretanja
    broj_pokretanja = ulazni_niz[0]
    global pop_vel
    pop_vel=ulazni_niz[1]
    global npop_vel
    npop_vel=pop_vel
    global mut_rate
    mut_rate=ulazni_niz[2]
    global uvodjenje_procenta_potomaka
    uvodjenje_procenta_potomaka=ulazni_niz[3]
    global rand_seed
    rand_seed=ulazni_niz[4]
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    global out_path
    out_path=ulazni_niz[5]
    global lower_bound
    lower_bound=ulazni_niz[6]
    global upper_bound
    upper_bound=ulazni_niz[7]
    global number_of_genes
    number_of_genes=ulazni_niz[8]
    global velicina_turnira
    velicina_turnira=ulazni_niz[9]
    global velicina_heuristickog_ukrstanja
    velicina_heuristickog_ukrstanja=ulazni_niz[10]
    #print(broj_pokretanja)
    if ispis:
        with open("cfg.txt", "w") as f:
            for line in range(len(ispis)):
                f.write(ispis[line]+":"+"\n")

def ucitavanje_putanje_ka_fajlu():
    putanja=input()
    return str(putanja)


def genetski():
    putanja=ucitavanje_putanje_ka_fajlu()
    ucitavanje_iz_fajla(putanja)
    # print(broj_pokretanja, pop_vel,
    #       npop_vel,
    #       mut_rate,
    #       uvodjenje_procenta_potomaka,
    #       rand_seed,
    #       out_path,
    #       lower_bound,
    #       upper_bound,
    #       number_of_genes,
    #       velicina_turnira,
    #       velicina_heuristickog_ukrstanja)
    outconsole =sys.stdout #open("GA-KXZK.txt", "a")
    print(out_path)
    outfile=open(out_path, "w")
    s_trosak = 0
    s_iteracija = 0
    best_ever_sol = None
    best_ever_f = None
    najbolji_lista = []
    srednja_lista = []
    broj_generacije=[]
    for k in range(broj_pokretanja):
        pop = population(pop_vel)
        print('Pokretanje GA za mutation rate:', mut_rate, "broj pokretanja", k+1,"populacija",pop_vel, file=outconsole)
        print('Pokretanje GA za mutation rate:', mut_rate,"broj pokretanja",  k+1,"populacija",pop_vel, file=outfile)
        najbolji_lista_generacijski = []
        srednja_lista_generacijski = []
        best = None
        best_f = None
        no_improv = 0
        t = 0
        while best_f != 0 and t < max_iter:
            n_pop = pop[:]
            while len(n_pop) < pop_vel + round(npop_vel*uvodjenje_procenta_potomaka):
                h1 = selection(pop, velicina_turnira)
                h2 = selection(pop, velicina_turnira)
                h3 = heuristic_crossover(h1, h2, 5)
                if h3[0] == 100:
                    continue
                if random.random() < mut_rate:
                    h3 = mutiraj(h3)
                n_pop.append(h3)
            pop = sorted(n_pop, key=lambda x: pomocna(x))[:pop_vel]
            f = pomocna(pop[0])
            trosak = sum(map(pomocna, pop)) / pop_vel
            srednja_lista_generacijski.append(trosak)
            t += 1
            if best_f is None or best_f > f:
                no_improv = 0
                best_f = f
                best = pop[0]
                najbolji_lista_generacijski.append(best_f)
            else:
                no_improv += 1
                najbolji_lista_generacijski.append(best_f)
                if no_improv == conv_numb:
                    break
            print("Najbolji u generaciji,", t, ": ",best_f,"Srednji trosak: ",trosak,file=outconsole)
            print("Najbolji u generaciji: ", t, ": ", best_f, "Srednji trosak: ", trosak, file=outfile)
        broj_generacije.append(t)
        najbolji_lista.append(najbolji_lista_generacijski)
        srednja_lista.append(srednja_lista_generacijski)
        s_trosak += best_f
        s_iteracija += t
        # ako smo našli bolji od prethodnog, ažuriramo najbolje rešenje
        if best_ever_f is None or best_ever_f > best_f:
            best_ever_f = best_f
            best_ever_sol = best
        print("Gotovo  ", k+1,". pokretanje algoritma  ","  Najmanji trosak:",best_f,"   Sastav hromozoma: ",best, file=outconsole)
        print(" ",file=outconsole)
        print("Gotovo  ", k + 1, ". pokretanje algoritma  ", "  Najmanji trosak:", best_f, "   Sastav hromozoma: ", best, file=outfile)
        print(" ", file=outfile)
    # na kraju svih izvršavanja izračunavamo srednji trošak i srednji broj iteracija
    s_trosak /= broj_pokretanja
    s_iteracija /= broj_pokretanja
    print("Posle svih iteracija")
    print('Srednji trosak: %.2f' % s_trosak, file=outfile)
    print('Srednji broj iteracija: %.2f' % s_iteracija, file=outfile)
    print('Najbolje resenje: %s' % best_ever_sol, file=outfile)
    print('Najbolji trosak: %.2f' % best_ever_f, file=outfile)
    print('Srednji trosak: %.2f' % s_trosak, file=outconsole)
    print('Srednji broj iteracija: %.2f' % s_iteracija, file=outconsole)
    print('Najbolje resenje: %s' % best_ever_sol, file=outconsole)
    print('Najbolji trosak: %.2f' % best_ever_f, file=outconsole)
    outfile.close()
    iscrtaj(najbolji_lista, srednja_lista, broj_generacije, pop_vel)


genetski()


