import numpy as np
import pandas as pd
import h5py

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from umap.umap_ import UMAP

if __name__ == '__main__':
    # parameters:
    data = 'train'   # test/train
    unit = 'residue'    # residue/protein
    subset = 'disorder'  # disorder/all


    embeddings_in = '../dataset/' + data + '_set.h5'
    embeddings = dict()
    with h5py.File(embeddings_in, 'r') as f:
        for key, embedding in f.items():
            original_id = embedding.attrs['original_id']
            embeddings[original_id] = np.array(embedding)
    # now ID and embeddings are written in the embeddings dictionary

    # Add embeddings to the labels and IDs, in correct order
    with open('../dataset/' + data + '_set_annotation.tsv', 'r') as labels:
        ids_ordered = list()
        labels_ordered = list()
        embeddings_ordered = list()
        disorder_residues = list()
        for i in labels.readlines()[1:]:
            tabs = i.split("\t")
            ids_ordered.append(tabs[0])
            embeddings_ordered.append(embeddings[tabs[0]])
            res = tabs[3].split(',')
            res = [int(i)-1 for i in res]
            disorder_residues.append(res)
            # create list of unique binding partner classes
            binding = sorted(list(set(tabs[4].split(','))))
            labels_ordered.append(','.join(binding))

    tsne_instance = TSNE(random_state=1, n_iter=250, metric='cosine')
    umap_instance = UMAP(n_neighbors=15, min_dist=0.3, metric='correlation')

    if subset == 'disorder':
        embeddings_selection = [embeddings_ordered[i][disorder_residues[i]] for i, _ in enumerate(ids_ordered)]
        subset_text = 'disorder_only'
    elif subset == 'all':
        embeddings_selection = embeddings_ordered
        subset_text = 'all_residues'
    else:
        raise ValueError('subset must be "all" or "disorder"')


    if unit == 'protein':
        # Average the Per-Residue embeddings to Protein embeddings
        clustering_input = [np.mean(x, axis=0) for x in embeddings_selection]
        unit_text_title = 'Protein Mean'
        unit_text = 'protein_mean'

    elif unit == 'residue':
        clustering_input = list()
        residue_label = list()
        residue_ID = list()
        for i, emb in enumerate(embeddings_selection):
            clustering_input.extend(emb)
            if subset == 'disorder':
                residue_label.extend([labels_ordered[i] for _ in emb])
            elif subset == 'all':
                for j, res in enumerate(emb):
                    if j in disorder_residues[i]:
                        residue_label.append(labels_ordered[i])
                    else:
                        residue_label.append('structured residue')
            residue_ID.extend([ids_ordered[i] for _ in emb])
        ids_ordered = residue_ID
        labels_ordered = residue_label
        unit_text_title = 'Residue'
        unit_text = 'residues'

    else:
        raise ValueError("unit must be 'protein' or 'residue'")


    # Run TSNE and UMAP
    print("started TSNE transformation")
    embs_tsne = tsne_instance.fit_transform(clustering_input)
    print("started UMAP transformation")
    embs_umap = umap_instance.fit_transform(clustering_input)
    print("done with transformations")

    # Extract X and Y columns from the TSNE and UMAP results
    embs_tsne_X = embs_tsne[:, 0]
    embs_tsne_Y = embs_tsne[:, 1]
    embs_umap_X = embs_umap[:, 0]
    embs_umap_Y = embs_umap[:, 1]

    df = pd.DataFrame(zip(embs_tsne_X, embs_tsne_Y, embs_umap_X, embs_umap_Y, ids_ordered, labels_ordered),
                      columns=['X_TSNE', 'Y_TSNE', 'X_UMAP', 'Y_UMAP', 'ID', 'Label'])

    # Plot the data frame
    sns.color_palette("flare", as_cmap=True)

    tsne_plot = sns.scatterplot(
        x="X_TSNE",
        y="Y_TSNE",
        hue="Label",
        data=df,
        legend="full",
        alpha=0.8,
        size=0.3,
        palette="deep"
    )
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title('T-SNE Clustering of ' + unit_text_title + ' Embeddings')
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig('../results/plots/' + data + '_tsne_' + unit_text + '_' + subset_text + '.png', bbox_inches="tight", dpi=600)
    plt.show()

    umap_plot = sns.scatterplot(
        x="X_UMAP",
        y="Y_UMAP",
        hue="Label",
        data=df,
        legend="full",
        alpha=0.8,
        size=0.3,
        palette="deep"
    )
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title('UMAP Clustering of ' + unit_text_title + ' Embeddings')
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig('../results/plots/' + data + '_umap_' + unit_text + '_' + subset_text + '.png', bbox_inches="tight", dpi=600)
    plt.show()