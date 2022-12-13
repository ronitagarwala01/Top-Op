"""
Framework:
Generation of new population
    DO NOT MAKE IT MORE COMPLICATED THAN NEEDED

    To wrapper function:
        Given dimensions and number of members to generate
        Call member generator:
            Member generator:
                Do:
                    Create array of given dimensions
                    Randomly populate each index of array with either 0 or 1
                Return array


Evaluation
    For each member within the population:
        Pass member to fitness function:
            Fitness function (element minimization):
                Sum the value of each array
                Return sum
    Pass sum and member to dtype packager:
        Dtype packager:
            dtype = [('member', object), ('sum', int)]
            return np.array(member, sum, dtype=dtype)



Selection
    Pass member/sum array to sorting function
        Sorting function:
            Sort array of member/sum by sum
            return array
    
    Pass elite ratio and members to elite selector
        Elite selector:
            Separate top ratio of memebers from the rest of the population
            return elite population, common population

    Pass cull ratio and common population to cull function
        Cull function:
            Given cull ratio, remove bottom % of population
            return resulting middle population

    Random pairing of members for crossover



Crossover
    Swapping of genetic information



Mutation
    Random chance of mutation, given by a probability
"""