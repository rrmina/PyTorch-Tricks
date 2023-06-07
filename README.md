# Reusable PyTorch codes

## Calculate Number of Parameters
```python
# Total Number of Parameters
total_params = sum(p.numel() for p in model.parameters())

# Total Number of Trainable Parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

## ImageFolder with Folder Paths

```python
class ImageFolderWithPaths(datasets.ImageFolder):
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]

        # make a new tuple that includes original and the path
        tuple_with_path = (*original_tuple, path)
        return tuple_with_path
```

## Reshape Layer for nn.Sequential

```python
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
```

## Debugger Layer for nn.Sequential
```python
class Debugger(nn.Module):
    def __init__(self):
        super(Debugger, self).__init__()
    
    def forward(self, x):
        print(x)
        return x
```

## Concatenate Images to Form Image Grids
```python
def concatenate_images(images, num_rows=2):
    B,C,H,W = images.shape
    num_cols = 1 + (B // num_rows) - (((B % num_rows) == 0) * 1)

    # Placeholder Image
    concat_image = np.zeros([C, num_rows*H, num_cols*W])

    # Make the Grid
    for i in range(B): 
        cur_row = i // num_cols
        cur_col = i % num_cols   
        concat_image[:, cur_row*H: (cur_row+1)*H, cur_col*W: (cur_col+1)*W] = images[i]

    # Remove Channel Dimension if Image assumed to be Graymap
    if (C==1):
        return concat_image.reshape(concat_image.shape[1], concat_image.shape[2])
    else:
        return concat_image
```

## Flexible Pyplot show image
```python
def show_img(img, title=""):
    # Check if image is a Graymap
    if (len(img.shape) == 2):
        H,W = img.shape
    else:
        C,H,W = img.shape
        img = img.transpose(1,2,0)

    fig = plt.figure(figsize=(10,10))
    plt.title(title)

    if (len(img.shape) == 2):
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)

    plt.show()
    plt.close()
```

## PGMDataset (ImageFolder specifically for Cropped Extended YaleB Dataset)
```python
class PGMDataset(Dataset):

    URL = "http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip"
    DATASET_FILE_NAME = "CroppedYale.zip"
    DATASET_FOLDER = "CroppedYale/"

    def __init__(self, dataset_path="data/", download=True, transform=None):
        self.dataset_path = dataset_path
        extended_dataset_path = dataset_path + PGMDataset.DATASET_FOLDER

        # Download Dataset
        if (download):
            self._download_one(PGMDataset.URL)

        # Extract Zip File
        if not os.path.exists(extended_dataset_path):
            self._extract_zip()

        self.image_paths = []
        self.label_map = {}

        count = 0
        # Get the list of image paths
        for folder in os.listdir(extended_dataset_path):
            class_folder = os.path.join(extended_dataset_path, folder)

            # Label - Integer map
            self.label_map[class_folder.split('/')[2]] = count

            for image_name in os.listdir(class_folder):

                # Skip non-PGM files. Some files .info
                if (image_name[-4:] != ".pgm"):
                    continue
                
                # Skip ambient files 
                if (image_name[-11:] != "Ambient.pgm"):
                    self.image_paths.append(os.path.join(class_folder, image_name))

            count += 1

        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        label = self.label_map[ self.image_paths[index].split('/')[2].split("\\")[0] ]

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image_paths)

    def _download_one(self, url):
        file_path = self.dataset_path
        zip_path = self.dataset_path + PGMDataset.DATASET_FILE_NAME

        # "data/"
        if not os.path.exists(file_path):
            os.makedirs( file_path )

        # "data/CroppedYale.zip"    
        if not os.path.exists( zip_path ):
            print( "Downloading ", PGMDataset.DATASET_FILE_NAME, " ...")

            with ur.urlopen(PGMDataset.URL) as response, open(zip_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)

    def _extract_zip(self):
        zip_path = self.dataset_path + PGMDataset.DATASET_FILE_NAME
        print("Extracting ", zip-path, " to ", self.dataset_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.dataset_path)
```
## Python Debugger for Multiprocess
```python
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    Source: https://stackoverflow.com/a/23654936
    
    To use:
        ForkedPdb().set_trace()

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
```

## Calling ipdb debugger durring error
```python
from ipdb import launch_ipdb_on_exception

with launch_ipdb_on_exception():
    ...
```

## Sorting a list according to another list (Method 1)
```python
x = [['A'], ['B', 'C'], ['D'], ['E']]
y = [1,0,3,2]

result_list = [i for _,i in sorted(zip(y, x))]
```

## Sorting a list according to another list (Method 2)
```python
x = [['A'], ['B', 'C'], ['D'], ['E']]
y = [1,0,3,2]

indices = [i for i in range(len(x))]                        # indices
sorted_indices = [idx for _,idx in sorted(zip(y,indices))]  # sorted indices
result_list = [x[idx] for idx in sorted_indices]            # sorted list
```

## Automatically making folders when saving file objects
```python
def safe_save(obj, filename):
    
    # Make sure the folder exists
    hierarchy = save_path.split("/")
    
    for i in range(1, len(hierarchy)):
        folder = "/".join(hierarchy[:i])
        
        if not os.path.exists(folder):
            os.mkdir(folder)
            
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
```

## Exporting Ipython Notebook to HTML with Attachments in Separate Folder
```
jupyter nbconvert --to html --ExtractOutputPreprocessor.enabled=True "notebook_name.ipynb"
```

## False Positive and/or False Negative Regularization
```python
#######################################################################################################
# Helper Function Only
# https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/4
# Converts [0,1,0,1] to [[1,0], [0,1], [1,0], [0,1]]
#######################################################################################################
def to_onehot(y, num_classes=2):
    y = y.view(-1, 1)
    batch_size = y.shape[0]
    y_onehot = torch.FloatTensor(batch_size, num_classes).to(y.device)
    y_onehot.zero_()

    y_onehot.scatter_(1, y, 1)

    return y_onehot
    
#######################################################################################################
#
#   FP_loss: Reduces false positives by lowering the logits that do not correspond to the correct label
# 
#   Example: 
#    Let y = label, y_oh = one-hot representation of y, and y_hat = logits
# 
#    For a Datapoint with y = [0] or y_oh [1, 0], and y_hat = [0.7, 0.6]
#       
#       A False positive happens if y_hat predicts [1], so to reduce the chance 
#       of predicting a False Positive, FP_loss aims to minimize the 2nd value of y_hat
#   
#    Similarly, for a datapoint with y = [1] or y_oh [0, 1], and y_hat = [0.7, 0.6]
#       FP_loss aims to reduce 0.7 so that after some time, 0.6 will be bigger than the 1st of y_hat
#
######################################################################################################
def FP_loss(logits, y):
    softmax_out = torch.softmax(logits, -1)
    max_softmax, _ = torch.max(softmax_out, dim=-1)
    softmax_normalised = softmax_out / max_softmax.view(-1,1)

    y_reversed = 1 - y
    y_onehot_reversed = to_onehot(y_reversed)

    fp_ = torch.sum(softmax_normalised * y_onehot_reversed, dim=1)
    fp_loss = torch.mean(fp_)

    return fp_loss
    
    
###################################################################### 
#                           Sample Usecase                           #
######################################################################
criterion = nn.CrossEntropyLoss()
classification_loss = critraion(out, y)             # Original Loss
fp_loss = FP_loss(out, y)                           # FP Loss Regularizer

BETA = 0.5                                          # Hyperparameter weight for FP loss, may use bigger weights
total_loss = classification_loss + BETA * fp_loss   # Combine losses

total_loss.backward()
optimizer.step()
```
