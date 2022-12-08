from doctest import UnexpectedException
from jinja2 import UndefinedError
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# import time
import pdb

# import collections
import pandas as pd
import scipy.constants as const # electric charge
import scipy.odr as odr
import scipy.optimize as opt
# import scipy.integrate.trapz as trapz


# from uncertainties import ufloat
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import AnyStr, Callable

# import os

from matplotlib.transforms import TransformedBbox
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase
from matplotlib._png import read_png





def prepare_frame_fixed(loc, dist_param = "distance",
):
"""
reads and processes fixed distance pmf file 

takes in the filename of the csv file, reads it in and 
takes the average at each distance
"""
    frame = pd.read_csv(loc)
    frame.dropna(inplace=True)

    frame = frame.sort_values(dist_param,ascending=True).reset_index().drop("index",axis=1)

    frame_mean = frame.groupby(dist_param, axis=0).mean().reset_index()
    return frame_mean

def prepare_frame_umbrella(loc, step_size):
    """
    takes spring csv file and creates preprocesses data

    The data is split in two. 
    df_group contains the average for all bins with the same r0 and springK
    df_weight contains all data for indiviual bins 

    """
        frame = pd.read_csv(loc)
        frame.dropna(inplace=True)

        frame["expo"] = np.exp(-BETA*frame["springK"]/2*(frame["distance"]-frame["effective_distance"])**2)
        # adding extra rows to perform operations on later
        # a newer pandas feature is available could make this unnecessary
        # but it was not available on my machine when i wrote this
        # so i fear, it might not be on yours eiterh

        frame["r0"] = frame["distance"]
        frame["dist_std"] = frame["effective_distance"]
        frame["dist_max"] = frame["effective_distance"]
        frame["dist_min"] = frame["effective_distance"]
        actual_bins = np.arange(min(frame.effective_distance),max(frame.effective_distance),step_size)
        center_bins = actual_bins[:-1]+step_size/2
        print("first center bins")
        print(center_bins)
        # print(np.unique(np.diff(center_bins*1e6), return_counts=True))
        bins = pd.cut(frame["effective_distance"],bins=actual_bins, labels=center_bins)    
        # df_check = frame.groupby(["distance", bins], axis=0).size().reset_index()
        df_weights = frame.groupby(["distance","springK", bins], axis=0).size().reset_index()
        print(f"len df_weiights: {len(df_weights)} vs len centerbins: {len(center_bins)}")
        # assert len(df_weights) == len(df_check), "denk over springK na"
        df_weights.columns = [*df_weights.columns[:-1], 'freqs']
        df_weights.rename(columns = {"effective_distance": "bins"})
        tot = np.zeros(len(df_weights))
        for ii, uniq in enumerate(np.unique(df_weights["distance"])):
            mask = (uniq == df_weights["distance"])
            tot[mask]=df_weights["freqs"][mask].sum()
        df_weights["probs"] = df_weights["freqs"]/tot
        df_weights.reset_index()

        frame["max_time"] = frame["time"]
        df_group = frame.groupby(["distance","springK"]).agg({
                "r0": "mean",
                "effective_distance": "mean",
                "dist_min": "min",
                "dist_max": "max",
                "dist_std": "std",
                "springM": "mean",
                "expo": "mean",
                "time": "count",
                "max_time": "max",
        }).rename(columns={"effective_distance": "dist_mean", "time": "freq"}).reset_index()
        total = df_group["freq"].sum()
        df_group["weights"] = df_group["freq"]/total
        print(center_bins.shape)
        print(np.min(center_bins))
        print(np.unique(np.diff(center_bins)))
        print(np.max(center_bins))
        return df_group, df_weights

def transform_umbrella(df_group, df_weights, is_bulk=False, return_weights=False):
    """
    calculates the pmf via umbrella integration

    umbrella integration uses gaussian approximates at every position

    as intermediate values (which can also be plotted for diagnostics)
    we calculate the force_chunks, the force at each distance and base distance (r0) 
    as well as probs, the probability distribution of a particle with a given r0 and
    spring constant (springK) and tot_weight the overall normalization weight for 
    each r0/springK pair.
    
    These are used to calulate the force at each distance and then the raw potential.
    Lastly a volume correction is applied
    """
    def gauss(x, *p):
        mu, sigma = p
        return 1/np.sqrt(2*np.pi)/sigma*np.exp(-(x-mu)**2/(2.*sigma**2))

    # convert actual bins into desired precision of chart 
    nrange = 500
    center_bins = np.unique(df_weights.effective_distance)
    distrange=np.linspace(np.min(center_bins),np.max(center_bins),nrange)

    # init intermediate value arrays
    force_chunks = np.zeros((len(df_group), nrange))
    probs = np.zeros((len(df_group), len(center_bins)))
    tot_weight = np.zeros(nrange)

    for ii, rr in enumerate(df_group.dist_mean):
        mask_bin = np.zeros_like(center_bins, bool)
        # isclose to avoid rounding error misses, yields 1 for same r0 and spring
        mask_r0 = ((np.isclose(df_weights["distance"],df_group.r0[ii])) & (np.isclose(df_weights["springK"],df_group.springK[ii])))
        for vals in df_weights["effective_distance"][mask_r0]:
            # mask bins = 1 if same bin is used
            mask_bin[(center_bins == vals)]=1

        window_weight =np.sum(df_weights["freqs"][mask_r0]) 
        probs[ii][mask_bin] = df_weights["freqs"][mask_r0]/window_weight
        weights = window_weight*gauss(distrange,df_group.dist_mean[ii],df_group.dist_std[ii])

        force_chunks[ii] = weights * (
            (distrange-df_group.dist_mean[ii]) /(BETA * df_group.dist_std[ii]**2)  
            - (distrange - df_group.r0[ii] ) * df_group.springK[ii] )
        tot_weight += weights
    
    force_chunks = np.divide(force_chunks, tot_weight, out=np.zeros_like(force_chunks), where=tot_weight!=0) # avoid dividing by zero
    force = np.sum(force_chunks, axis=0)
    potentials = make_pot(distrange,-force)

    
    # apply correction for shape of box 
    potentials += pot_shift(distrange, is_bulk)

    output_dict = {
        "distrange": distrange,
        "potentials": potentials,
        "tot_weight": tot_weight,
        "force_chunks": force_chunks,
        "probs": probs,
        "springK" : df_group.springK.to_numpy(),
    }

    return output_dict

def pot_shift(distrange, is_bulk):
    """
    calculates correction of potential due to shape of simulation box

    in the bulk case, we have spheres for effectively all distances (i.e. smaller than half L) 
    in the slit case, spheres exist only for very short distances. 
    Afterwards, we get cylinders.
    The contribution of this volume is then calculated and returned
    """
    maxcube = 25 if is_bulk else 5 # max length, either half z or half 
    volmask = distrange<maxcube 
    uprange = distrange+(distrange[1]-distrange[0])/2
    downrange = distrange-(distrange[1]-distrange[0])/2
    volume =  4/3*np.pi*(uprange**3-downrange**3)
    area = 2*np.pi*(uprange**2-downrange**2)*maxcube
    space = area
    space[volmask]=volume[volmask]
    return 1/BETA * np.log(space/space[-1])


def load_data(filename, 
    is_fixed=False, fresh=False, step_size=0,
    folder="/home/cvaz/UU/thesis/csv",
    ):
    """
    wrapper function to avoid reprocessing of same data

    tries to load filename from stored file, if unsuccessful 
    or forced via fresh==True, loads data via function
    
    if is_fixed: 
        treats file as constrained bias frame 
        and returns one dataframe
    if not: 
        treats file as umbrella sampled and returns two frames
        forces the choice of bin size
    """

    if not is_fixed: assert step_size>0, "choose step_size"
    loc=f'{folder}/{filename}'
    print("retrieving file from:")
    print(loc)
    base_name = loc[:-4] 
    ending = ".pkl"
    if is_fixed :
        names = [base_name + "-mean" + ending]
    else:
        names   = [ base_name + f"-s{step_size}-group"  + ending,
                    base_name + f"-s{step_size}-weight" + ending ]
    try:
        if fresh: 
            raise FileNotFoundError
        frames = []
        for name in names:
            frames.append(pd.read_pickle(name))
    except FileNotFoundError:
        print('no existing file')
        if is_fixed:
            frames = [prepare_frame_fixed(loc)]
        else:
            frames = prepare_frame_umbrella(loc, step_size=step_size)
        for frame,name in zip(frames,names):
            pd.to_pickle(frame, name)
    else: 
        print("reovered successfully")
    return frames

def make_pot(distances, diff):
    """
    integrates over forces from dist[x] to dist[max]
    """
    potentials=[]
    for ii in range(len(diff)):
        # pot = np.sum(df_dist[c][ii:])
        pot= np.trapz(diff[ii:], x=distances[ii:])
        potentials.append(pot)
    return np.array(potentials)

def make_pot_fix(frame_mean, dist_param = "distance",
    col_bas= "freezeBy", col_mov ="freezeMy" ,
    ):
    """
    calculates the energy at a given distance using fixed pmf
    """

    distances = frame_mean[dist_param]
    diff = np.array(frame_mean[col_mov]-frame_mean[col_bas])/2
    return np.array(distances), make_pot(distances, diff)

    
def write_energy(loc, dist, pot, firstline):
    """
    write energy to file, 
    with name according to input file as txt

    file format:
    dist1   pot1
    dist2   pot2
    dist3   pot3
    """
    base_name = loc[:-4]
    ending = ".txt"
    secondline = "distance potential"
    np.savetxt(base_name+ending, np.transpose((dist,pot)), 
                 delimiter="\t", header= f"{firstline}\n{secondline}")

def get_energy_fix(filename, energy2file=True, ):
    """
    enter data frame of fixed position pmf
    and obtain data as object or file
    """
    frame_mean = load_data(filename,is_fixed=True, fresh=True)[0]
    dist, pot = make_pot_fix(frame_mean)
    comment=f"Constrained Bias calculations from {filename}"
    if  energy2file: write_energy(filename, dist, pot, comment)
    return dist, pot


def get_energy_ui( filename,    step_size=0.01,  fresh=False, is_bulk=False, energy2file=True):
    """
    enter data frame of umbrella sampling run 
    and obtain pmf via umbrella integration
    with the data returned as an object or saved to file
    """
    frames = load_data(filename, step_size=step_size, fresh=True)
    df_group, df_weights = frames[0],frames[1]
    output_dict = transform_umbrella(df_group, df_weights, is_bulk=is_bulk)  

    dist, pot = output_dict["distrange"], output_dict["potentials"]
    comment=f"Umbrella Integration calculations from {filename}"
    if  energy2file: write_energy(filename, dist, pot, comment)
    return dist, pot


TEST_FIX = "fix-bulk-licl.csv"
TEST_SPRING = "spring-caso-K150-s5.csv"
BETA = 1/0.593 #in units of kcal/A/mol

def test():
    get_energy_ui(TEST_SPRING)    
    get_energy_fix(TEST_FIX)    

test()
