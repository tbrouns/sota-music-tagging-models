def get_pickle_filename(split, category):
    if category is not None:
        filename = f"{category}_{split}.pkl"
    else:
        filename = f"{split}.pkl"
    return filename
