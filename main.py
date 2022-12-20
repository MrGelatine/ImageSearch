import numpy as np
import cv2 as cv
from os import walk
from sklearn.cluster import KMeans
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import PIL
import torch
import clip
import faiss
from PIL import Image
from GPUtil import showUtilization as gpu_usage
from numba import cuda


def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()
def getDescriptors(img):
    img = np.array(img)[:, :, ::-1].copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    _, dscpr = sift.detectAndCompute(gray, None)
    return dscpr
def getTrainData(trainPath,ceil):
    result = np.empty((0, 128))
    cnt = 0
    for (dirpath, dirnames, filenames) in walk(trainPath):
        for img_path in filenames:
            img = cv.imread(f"{dirpath}\{img_path}")
            result = np.append(result,getDescriptors(img),axis=0)
            cnt += 1
            if(cnt == ceil):
                break
    print("Data loading is ok!")
    return result
def generateDescriptorClassifier(dataPath, savePath, clusters_n=1000, iterations=4, ceil=100):
    data = getTrainData(dataPath, ceil)
    print("PCA is ok!")
    kmeans = KMeans(init="k-means++", n_clusters=clusters_n, n_init=iterations)
    print("Init is ok!")
    print(data.shape)
    kmeans.fit(data)
    print("Done!")
    with open(savePath, 'wb') as handle:
        pickle.dump(kmeans,handle)
def getImageEmbedding(img, classifier_path, categories=512):
    with open(classifier_path, 'rb') as handle:
        classifier = pickle.load(handle)
    descr = getDescriptors(img)
    if descr is not None:
        classified_descr = classifier.predict(descr.astype(float))
        unique, counts = np.unique(classified_descr, return_counts=True)
        result = np.zeros(categories)
        result[unique] = counts#/counts.sum()
        return result.astype('float32')
    else:
        return -1
def fillDataCSV(csv_path, data_path, classifier_path, categories=512, max_exmpl=5000, model= None):
    files = []
    embeddings = []
    true_embeddings = []
    c = 1
    for (dirpath, dirnames, filenames) in walk(data_path):
        for img_path in filenames:
            if model is None:

                img = cv.imread(f"{dirpath}\{img_path}")
                embed = getImageEmbedding(img, classifier_path,categories)
            else:
                with torch.no_grad():
                    img = preprocess(Image.open(f"{dirpath}\{img_path}")).unsqueeze(0).cuda()
                    embed = model.encode_image(img).cpu().squeeze()
            if not isinstance(embed, int):
                files.append(f"{dirpath}/{img_path}")
                embeddings.append(' '.join(str(x) for x in embed.tolist()))
                true_embeddings.append(np.asarray(embed).astype('float32'))
                del embed
                del img
                torch.cuda.empty_cache()
                #print("After:")
                #print(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
                #print()

            print(f"{c}/{len(filenames)}")
            c += 1
            if(max_exmpl == c):
                break
    true_embeddings = np.asarray(true_embeddings).astype('float32')
    index = faiss.IndexFlatL2(categories)
    faiss.normalize_L2(true_embeddings)
    index.add(true_embeddings)
    faiss.write_index(index, f"{csv_path[:-4]}.index")
    print(index)
    d = {'file_path': files, 'embeddings': embeddings}
    df = pd.DataFrame(data=d)
    df.to_csv(csv_path)
def matchImage(img, csv_path, classifier_path, categories=512, model = None):
    data = pd.read_csv(csv_path)
    if (model is None):
        embed = getImageEmbedding(img, classifier_path, categories).reshape(1, -1)
    else:
        tens_img = preprocess(img).unsqueeze(0).cuda()
        embed = model.encode_image(tens_img).cpu().squeeze()
    index = faiss.read_index(f"{csv_path[:-4]}.index")
    embed = np.asarray(embed.unsqueeze(0),dtype="float32")
    D, I = index.search(embed, 5)
    I = np.squeeze(I,axis= 0)
    D = np.squeeze(D, axis=0)
    img_path_0 = data.iloc[[I[0]]]['file_path'].iloc[0]
    img_path_1 = data.iloc[[I[1]]]['file_path'].iloc[0]
    img_path_2 = data.iloc[[I[2]]]['file_path'].iloc[0]
    img_path_3 = data.iloc[[I[3]]]['file_path'].iloc[0]
    img_path_4 = data.iloc[[I[4]]]['file_path'].iloc[0]
    st.image(img, caption=f'Original')
    st.image(Image.open(img_path_0), caption=f"Absolute same {D[0]}")
    st.image(Image.open(img_path_1), caption=f"Same {D[1]}")
    st.image(Image.open(img_path_2), caption=f"Almost same {D[2]}")
    st.image(Image.open(img_path_3), caption=f"Little same {D[3]}")
    st.image(Image.open(img_path_4), caption=f"No same {D[4]}")
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    #generateDescriptorClassifier("VOC/data",'voc_desc_clsrt2048.pickle',1024,200)
    fillDataCSV("VOC/voc_512_nn.csv", "VOC/data", "classifier/desc_clsrt512.pickle",512,10000, model)
    #fillDataCSV("VOC/clip_voc_1000.csv", "VOC/data", "classifier/desc_clsrt1024.pickle", 1024, 1000, model)
    #fillDataCSV("VOC/voc_2048.csv", "VOC/data", "classifier/desc_clsrt2048.pickle", 2048)
    #img = Image.open("COCO/data/000000000650.jpg")#cv.imread("COCO/000000000650.jpg")
    #matchImage(img, "VOC/voc_512_nn.csv", "classifier/desc_clsrt1024.pickle",512, model)
    #img = cv.imread("VOC/data/2007_002470.jpg")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp", "tiff"])
    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file).convert("RGB")
        image = PIL.ImageOps.exif_transpose(image)
        matchImage(image, "VOC/voc_512_nn.csv", "classifier/desc_clsrt512.pickle", 512, model)
