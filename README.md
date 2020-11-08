Będzie gotowe za tydzień. Informacje o projekcie w info.pdf

# Szukanie środka twarzy 

_Author: Michał Janik_

Finding the center of the face in the photo

## Install requirements

```bash
pip install -r requirements.txt
```

## Download dataset(CelebA)

The dataset may be downloaded from [here][celeba].

Extract `img_celeba.7z` to `data/img_celeba` and put file `list_landmarks_celeba.txt` everything in `data/Anno`.

Your directory should look like this:
```
data/
    Anno/
      list_landmarks_celeba.txt
    img_celeba/
      000001.jpg
      000002.jpg
      000003.jpg
      ...
```

## Build dataset

Run the script `build_dataset.py` which will generate file 'data.csv' containing image paths and target labels (nose_scaled_x,nose_scaled_y):

```bash
python build_dataset.py
```

## Train Model
In order to **train** the model, run:

```
python train_model.py
```

## Evaluate Model
To see how the model performs on the test data, run:
```
python evaluate_model.py
```

[celeba]: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
