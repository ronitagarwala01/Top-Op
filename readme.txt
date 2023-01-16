This branch is dedicated to generating a dataset for the ML model. This dataset will focus on calculating compliance.

Each sample will contain the following items:
    - The supports/anchors of the part.
    - The forces being applied to the part.
    - The current part as a density grid.
    - The stiffness matrix of the part calculated from the current part.
    - The displacement vector of the part at its current state when the given forces are applied.
    - The compliance of the part when under forces.
    - The jacobian of the compliance of the part under the given forces.

Samples will be generated as follows.
    - A random assortment of possible problem configurations will be generated(cicles with load conditions)
    - A genetic algorithm will be run to find the minimum mass for each configuration.
        - This GA will have a low population count.
    - For each agent in the GA, a file will be created and all the above components from the dataset will be saved to the file.

Extra data can be generated using the following methods:
    - If we have the the parts current density grid x, along with its compliance, we can generated a new data point by adding noise light to x and saving just x the forces, the supports, and the final compliance
        - This method will create more data points at the cost of being lightly inaccurate data points.