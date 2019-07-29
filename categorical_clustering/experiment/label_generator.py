import random


def generate_labels(labels_dict):
    labels = []
    for key in labels_dict.keys():
        labels += [key] * labels_dict[key]
    random.shuffle(labels)
    return labels
