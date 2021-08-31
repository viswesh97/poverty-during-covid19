import seaborn as sns
import matplotlib.pyplot as plt
from config import *

def plots():
    fig, ax = plt.subplots(figsize = (20,10))
    sns.barplot(x=data["Multidimensional Poverty Index\n (MPI = H*A)"],
             y=data["Country"],
             data=data,ci=None,ax=ax)
    plt.xlabel("MPI")
    plt.ylabel("Country")
    plt.show()


    fig, ax = plt.subplots(figsize = (40,10))
    sns.lineplot(x=data['GDP'],
             y=data['Multidimensional Poverty Index\n (MPI = H*A)'],
             data=data,sort=True,ax=ax)
    plt.xlabel("GDP")
    plt.ylabel("MPI")
    plt.show()

    fig, ax = plt.subplots(figsize = (20,10))
    sns.barplot(x=data["Child Mortality"],
             y=data["Multidimensional Poverty Index\n (MPI = H*A)"],
             data=data,ax=ax)
    plt.xlabel("Child Mortality")
    plt.ylabel("MPI")
    plt.show()

    fig, ax = plt.subplots(figsize = (20,10))
    sns.scatterplot(x=data["World Region"],
                 y=data["Country"],
                 hue=data['Poverty'],data=data,ax=ax)
    plt.xlabel("World Region")
    plt.ylabel("Country")
    plt.show()