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
