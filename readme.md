#Compliance Minimization Dataset

This branch if for generating and experimenting with the compliance minimization algorithm.

##Data
In order to get the max amount of data from the topopt we have split the datasets into two classes. 
Each data point has saved in a seperate file the load conditions of the problem as well as the iteration data.

The load condtions stores all the information needed to recreate the problem in the topopt.
The iterations is the actual iteration that is created in order to minimize the compliance of the element
There is only one set of load conditions for each datapoint where as the iteration data may range from 10 to 50 iterations

The data created by the TopOpt contains the following information for the initial load conditions:
    - forces: forces applied to the part
    - free: the specific degrees of freedom that the part can move along(all the elements that are not anchored)
    - passive: The areas of the part where it must remain solid or empty
    - volfrac: The allowable volume of the part
    - nelx: the number of elements along the x axis
    - nely: the number of elements along the y axis
    - penal: The compliance penalty(should always be 3)
    - rmin: The filter radius of the part(should always be 5.4 for no particular reason)

The Data for each iteration is as follows:
    - x: the current part 
    - xPhys:  the current part after filtering
        - The first xPhys will just be a solid block of volfrac
    - compliance: the compliance of xphys
    - change: the recorded change between the previous and current xphys
    - mass: the current mass of xphys
        - Since the compliance minimization is set with a volume constraint then this value should not change
    