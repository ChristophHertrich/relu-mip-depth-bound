# This SageMath script solves the mixed-integer program described in

# Christoph Hertrich, Amitabh Basu, Marco Di Summa, Martin Skutella:
# Towards Lower Bounds on the Depth of ReLU Neural Networks,

# proving that no H-conforming 3-layer NN can compute the maximum of five numbers.

import itertools as it
import datetime

# set debug to True in order to print some intermediate results
# keep it False if you don't want to dive into the code details
debug = False

# helper function to keep track of the runtime:
def print_time():
    now = datetime.datetime.now()
    print("Current date and time: ")
    print(str(now))

# precompute a list of all non-empty, proper subsets of {0,1,2,3,4},
# corresponding to the 30 rays of the hyperplane arrangement H:
subsets = []
for cardinality in range(1,5):
    subsets += list(it.combinations(range(5),cardinality))
if debug:
    print("Subsets: ")
    print(subsets)
    print()
    
# compute the objective function coefficients as (-1)^|S|:
obj = []
for subset in subsets:
    obj.append((-1)**len(subset))
if debug:
    print("Objective function coefficients: ")
    print(obj)
    print()
    
# get the corresonding list of the rays:
rays = []
for subset in subsets:
    ray = [0]*4
    for index in subset:
        if index == 0:
            ray = [1]*4
        else:
            ray[index-1] -= 1
    rays.append(ray)
if debug:
    print("Rays: ")
    print(rays)
    print()
    
# for each ray get its function values
# for all 14 basis functions in B^14:
values_at_rays = []
for ray in rays:
    values_at_ray = list(ray)
    for i in range(4):
        values_at_ray.append(max(0, ray[i]))
    for (i,j) in it.combinations(range(4),2):
        values_at_ray.append(max(ray[i], ray[j]))
    values_at_rays.append(values_at_ray)
if debug:
    print("Values at rays: ")
    for values_at_ray in values_at_rays:
        print(values_at_ray)
    print()

# Determine all pairs of indices such that one of the
# corresponding subsets is contained in the other:
# (need to check only one direction because our subsets
# are sorted by cardinality)
subset_pairs = []
for pair in it.combinations(range(30), 2):
    if set(subsets[pair[0]]).issubset(set(subsets[pair[1]])):
        subset_pairs.append(pair)
if debug:
    print("Number of subset pairs: " + str(len(subset_pairs)))
    print("Subset pairs:")
    for pair in subset_pairs:
        print(str(subsets[pair[0]]) + " - " + str(subsets[pair[1]]))
    print()
    
    
# Finally define the MIP. Solver ppl is used for exact arithmetics
mip = MixedIntegerLinearProgram(solver="ppl", maximization=True)

# Define the variables as stated in the paper:
a = mip.new_variable(real=True, nonnegative=False, indices=range(14), name="a")
mip.set_min(a,-1)
mip.set_max(a,1)
z = mip.new_variable(binary=True, indices=range(30), name="z")
y = mip.new_variable(real=True, nonnegative=True, indices=range(30), name="y")

# Define a linear expression for the activation at the i-th ray (g(r_S) in the paper):
activation = []
for i in range(30):
    activation.append(mip.sum([values_at_rays[i][j] * a[j] for j in range(14)]))

# Constraints to ensure correct ReLU computation at each ray:
# (The y[i] >= 0 constraint is already captured by variable initialization above)
for i in range(30):
    mip.add_constraint(y[i]>=activation[i])
    mip.add_constraint(y[i]<=15*z[i])
    mip.add_constraint(y[i]<=activation[i] + 15*(1-z[i]))
    
# Constraints to ensure that the output function is H-conforming:
for pair in subset_pairs:
    mip.add_constraint(activation[pair[0]] >= 15*(z[pair[1]]-1))
    mip.add_constraint(activation[pair[1]] >= 15*(z[pair[0]]-1))

# Setting the MIP objective function:
mip.set_objective(sum([obj[i] * y[i] for i in range(30)]))

# In debug mode, output the base ring, in order to verify exact rational arithmetics:
if debug:
    print("MIP base ring:")
    print(mip.base_ring())
    print()

# Solving the MIP:
print("Start MIP solving ...")
print_time()
mip.solve()
print("MIP solved!")
print_time()
print("Objective Value:")
print(mip.get_objective_value())
print()

# printing the coefficients in the optimum solution (in debug mode):
if debug:
    print("Optimum Coefficients:")
    for i, v in sorted(mip.get_values(a).items()):
        print('a_%s = %s' % (i, v))
    print()
