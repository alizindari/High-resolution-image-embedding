# High-resolution-image-embedding-using-GNN
A method based on Graph Attention NN for image embedding and classification

## Image Cropping
In this step the high resolution image shoud be cropped to small 224 * 224 image. so each image is converted to a batch of small images.
![Alt text](https://github.com/alizindari/High-resolution-image-embedding/blob/main/images/cropping.PNG "Image Cropping")

## Feature Extraction
In this step all of the small images will pass through a backbone network such as ResNet50 which is used in this work to obtaion feature vectors for each small image.
![Alt text](https://github.com/alizindari/High-resolution-image-embedding/blob/main/images/features.PNG "Feature Extraction")

## Converting High Resolution Image to Graph
By obtaining the features from next part now we should convert these small images to a graph. I did that by addng edge between those images which have a cosine similarity less than 0.2. After this part the each high resolution image is converted to a graph.
![Alt text](https://github.com/alizindari/High-resolution-image-embedding/blob/main/images/convert_to_graph.PNG "Image to Graph")

## Graph Attention Network and Graph Pooling
By converting each image to a graph now we can perform graph operation on the image. In this work the goal is to find an embedding for each image so I used the lables for the classification task and perfomed graph classification algorithm to obtain the final embedding.
![Alt text](https://github.com/alizindari/High-resolution-image-embedding/blob/main/images/graph.png "Graph Pooling")


## Dataset
The dataset is from FloodNet challenge that can be accessed from <a href="http://www.classic.grss-ieee.org/earthvision2021/challenge.html" target="_top">Here</a>
