import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List
from scipy.cluster.vq import kmeans, vq, whiten
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes

def prompt_select(df : pd.DataFrame) -> Dict[str, List[str]]:
    obj_res = {'categorical' : [], 'numerical' : []}
    num_cols = len(df.columns)
    df_t = df.T
    df_t.insert(0, 'INDEX', range(0, len(df_t)))

    print("Here is a sample of your dataframe. It's been rotated for printing purposes, don't worry.")
    print(df_t.iloc[0:num_cols, 0:3])
    print("\nPlease do the following:")
    print("o) Write the index of the header, followed by a ',' (comma, without spaces), and its type (cat, for categorical or num, for numerical) for the columns you want to consider.\nGood luck!\n\n")

    while True:
        x = input(" > ")
        if not x:
            break
        temp = x.split(',')
        if(len(temp) != 2):
            print("Invalid format. Please write in the format {i_header},{num|cat}")
            continue
        if (int(temp[0]) < len(df.columns)):
            if (temp[1] == 'cat'):
                obj_res['categorical'].append(df.columns[int(temp[0])])
            elif (temp[1] == 'num'):
                obj_res['numerical'].append(df.columns[int(temp[0])])
            else:
                print("Invalid type. Try again.")
        else:
            print("Invalid header index. Try again.")

    return (obj_res)

def exec_kmeans(df, choices_obj):
    print("Whitening data...", end='', flush=True)
    for header in choices_obj['numerical']:
        df[header + "_scaled"] = whiten(df[header])
    print("Done.")
    k = int(input("Number of clusters:\n > "))

    headers_choices = [header + "_scaled" for header in choices_obj['numerical']]
    cluster_centers, distortion = kmeans(df[headers_choices], k)
    df['cluster_labels'], distortion_list = vq(df[headers_choices], cluster_centers)

    # Plot clusters
    print("Only showing 2 dimensions of data (picking first two headers)")
    sns.scatterplot(x=headers_choices[0], y=headers_choices[1],
                    hue='cluster_labels', data = df)
    plt.show()

def exec_kmodes(df, choices_obj):
    # reproduce results on small soybean data set
    cats_not_scaled = [header for header in choices_obj['categorical']]

    X = df[cats_not_scaled].astype(str)
    k = int(input("Number of clusters:\n > "))

    kmodes_cao = KModes(n_clusters=k, init='Cao', verbose=1)
    kmodes_cao.fit(X.values)

    # Print cluster centroids of the trained model.
    print('k-modes (Cao) centroids:')
    print(kmodes_cao.cluster_centroids_)
    # Print training statistics
    print('Final training cost: {}'.format(kmodes_cao.cost_))
    print('Training iterations: {}'.format(kmodes_cao.n_iter_))

def exec_kprototypes(df, choices_obj):

    print("Whitening data...", end='', flush=True)
    for header in choices_obj['numerical']:
        df[header + "_scaled"] = whiten(df[header])
    print("Done.")
    nums_scaled = [header + "_scaled" for header in choices_obj['numerical']]
    cats_not_scaled = [header for header in choices_obj['categorical']]

    X = pd.concat([df[nums_scaled].astype(float), df[cats_not_scaled].astype(str)], axis=1)
    k = int(input("Number of clusters:\n > "))
    kproto = KPrototypes(n_clusters=k, init='Cao', verbose=2)

    df['cluster_labels'] = kproto.fit_predict(X.values, categorical=list(range(len(X.columns) - len(cats_not_scaled), len(X.columns))))
    if(len(nums_scaled) >= 2):
        # Plot clusters
        print("Only showing 2 dimensions of data (picking first two headers)")
        sns.scatterplot(x=nums_scaled[0], y=nums_scaled[1],
                        hue='cluster_labels', data = df)
        plt.show()

def main():
    df = pd.read_csv('datasets/sales_data_sample.csv', encoding = 'ISO-8859-1')

    while(True):
        file_name = input("Do you have a config file? Write it's name (we'll look for it inside datasets/) if you do, press Enter if not.\n >  ")
        if not file_name:
            choices_obj = prompt_select(df)
            print(json.dumps(choices_obj, indent=4))
            if (input("Is this tagging ok? y/n\n > ") == 'y'):
                break
            print("Let's try again.")
        try:
            with open('datasets/' + file_name, encoding='utf-8') as data_file:
                choices_obj = json.loads(data_file.read())
            print(json.dumps(choices_obj, indent=4))
            if (input("Is this tagging ok? y/n\n > ") == 'y'):
                break
            print("Let's try again.")
        except Exception as e:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(e).__name__, e.args)
                print (message)

    if (choices_obj['categorical'] != [] and choices_obj['numerical'] != []):
        # k-prototypes
        exec_kprototypes(df, choices_obj)
        print("k-prototypes")
    elif (choices_obj['categorical'] == [] and choices_obj['numerical'] != []):
        # k-means
        exec_kmeans(df, choices_obj)
        print("k-means")
    elif (choices_obj['categorical'] != [] and choices_obj['numerical'] == []):
        exec_kmodes(df, choices_obj)
        print("k-modes")
    else:
        # nothing selected
        print("Nothing was selected")


if __name__ == '__main__':
    main()
