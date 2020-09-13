import splitfolders




DATA_PATH = "/Users/sb/tensorflow_datasets/caltech_birds2010_6000" 
SEED = 0



# Split with a ratio.
splitfolders.ratio(
    
    DATA_PATH, output=DATA_PATH + "_split_RATIO", 
    seed=SEED, ratio=(.8, .1, .1), group_prefix=None
    
    # To only split into training and validation set, 
    #  set a tuple to `ratio`, i.e, `(.8, .2)`.
)
    

# # Split val/test with a fixed number of items e.g. 100 for each set.
# splitfolders.fixed(
    
#     # Source/target directories and seed
#     DATA_PATH, output=DATA_PATH + "_split_FIXED", 
#     seed=SEED, fixed=(100, 100), 
#     oversample=False, group_prefix=None
    
#     # To only split into training and validation set, 
#     #  use a single number to `fixed`, i.e., `10`.   
# )